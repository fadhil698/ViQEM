"""
================================================================================
qem_pipeline.py - Unified ML-QEM Pipeline (Consolidated)
================================================================================

Satu file tunggal yang menggabungkan:
  - qem_main_updated_v2.py  (core components)
  - pipeline.py             (ExtendedPipeline, VQEPipeline)
  - UnifiedPipeline.py      (UnifiedPipeline)

Semua nama kelas dan fungsi dipertahankan agar kompatibel dengan kode lain.

Struktur:
  0. H2HamiltonianGenerator
  1. AnsatzConfig + GenerateAnsatz
  2. EnergyResult + CalculateEnergy
  3. TrainML
  4. VQERunner
  5. ExtendedPipeline
  6. VQEPipeline
  7. UnifiedPipeline
  8. Utility: create_h2_hamiltonian()
  9. main()
================================================================================
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import n_local
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeBelemV2, FakeAthensV2

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper

from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import NumPyMinimumEigensolver

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from custom_noise import CustomNoiseBuilder  # noqa: F401 – kept for compatibility

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 0. H2 HAMILTONIAN GENERATOR
# ============================================================================

class H2HamiltonianGenerator:
    """Generate qubit Hamiltonian untuk molekul H2 dengan variable bond length."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        logger.info(f"H2HamiltonianGenerator initialized: {n_qubits} qubits")

    def generate_h2_hamiltonian(self, bond_length: float) -> SparsePauliOp:
        """Generate H2 Hamiltonian untuk bond length tertentu (Angstrom)."""
        driver = PySCFDriver(
            atom=f"H 0 0 0; H 0 0 {bond_length}",
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        es_problem = driver.run()
        second_q_op = es_problem.hamiltonian.second_q_op()

        if self.n_qubits == 4:
            mapper = JordanWignerMapper()
        elif self.n_qubits == 2:
            mapper = ParityMapper(num_particles=es_problem.num_particles)
        else:
            raise ValueError(f"Unsupported n_qubits: {self.n_qubits}")

        return mapper.map(second_q_op)

    def generate_hamiltonian_range(
        self, bond_lengths: np.ndarray
    ) -> Dict[float, SparsePauliOp]:
        """Generate Hamiltonian untuk setiap bond length dalam array."""
        hamiltonians = {float(r): self.generate_h2_hamiltonian(r) for r in bond_lengths}
        logger.info(f"✓ Generated {len(hamiltonians)} Hamiltonians")
        return hamiltonians


# ============================================================================
# 1. ANSATZ GENERATION
# ============================================================================

@dataclass
class AnsatzConfig:
    """Configuration untuk ansatz generation."""
    n_qubits: int = 4
    rotation_blocks: List[str] = None          # default diset di __post_init__
    entanglement_blocks: str = "cx"
    entanglement: str = "linear"
    reps: int = 2
    insert_barriers: bool = True
    param_range: Tuple[float, float] = (-np.pi, np.pi)

    def __post_init__(self):
        if self.rotation_blocks is None:
            self.rotation_blocks = ["ry"]


class GenerateAnsatz:
    """Membuat quantum circuits dengan parameterized ansatz."""

    def __init__(self, config: AnsatzConfig):
        self.config = config
        self.ansatz: Optional[QuantumCircuit] = None
        logger.info(f"Ansatz config: {config.n_qubits} qubits, reps={config.reps}")

    def create_ansatz(self) -> QuantumCircuit:
        """Buat ansatz circuit dengan NLocal."""
        self.ansatz = n_local(
            self.config.n_qubits,
            rotation_blocks=self.config.rotation_blocks,   # <-- dulu hardcoded ["ry"]
            entanglement_blocks=self.config.entanglement_blocks,
            entanglement=self.config.entanglement,
            reps=self.config.reps,
            insert_barriers=self.config.insert_barriers,
        )
        logger.info(
            f"✓ Ansatz created: {self.ansatz.num_parameters} params | "
            f"rot={self.config.rotation_blocks}, ent={self.config.entanglement}/"
            f"{self.config.entanglement_blocks}, reps={self.config.reps}"
        )
        return self.ansatz

    def get_info(self) -> Dict:
        """Get ansatz information."""
        if self.ansatz is None:
            self.create_ansatz()
        return {
            "n_qubits": self.config.n_qubits,
            "n_parameters": self.ansatz.num_parameters,
            "depth": self.ansatz.decompose().depth(),
        }


# ============================================================================
# 2. ENERGY CALCULATION (Ideal, Noisy, ZNE)
# ============================================================================

@dataclass
class EnergyResult:
    """Hasil perhitungan energi."""
    ideal_energy: Optional[float] = None
    noisy_energy: Optional[float] = None
    zne_energy: Optional[float] = None
    theta_opt: Optional[np.ndarray] = None
    backend_name: str = "FakeBelemV2"
    noise_model: Optional[str] = None


class CalculateEnergy:
    """Hitung energi dengan multiple backends: ideal, noisy, ZNE."""

    def __init__(
        self,
        n_qubits: int = 4,
        backend_class=FakeAthensV2,
        noise_model: Optional[NoiseModel] = None,
    ):
        self.n_qubits = n_qubits
        self.backend_class = backend_class

        # Ideal estimator (no noise)
        self.ideal_estimator = Estimator(options={"default_precision": 1e-2})

        # Noisy estimator – use provided noise_model or derive from fake backend
        self.backend_fake = backend_class()
        effective_noise = noise_model if noise_model is not None else NoiseModel.from_backend(self.backend_fake)
        self.noisy_estimator = Estimator(
            options={
                "default_precision": 1e-2,
                "backend_options": {"noise_model": effective_noise},
            }
        )
        logger.info("Energy calculators initialized: ideal + noisy + ZNE")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def calculate_groundstate(self, hamiltonian: SparsePauliOp) -> float:
        """Exact ground-state energy via NumPyMinimumEigensolver."""
        result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(hamiltonian)
        energy = float(result.eigenvalue.real)
        logger.info(f"✓ Exact ground truth energy: {energy:.6f}")
        return energy

    def calculate_ideal_energy(
        self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, theta: np.ndarray
    ) -> float:
        """Ideal energy (tanpa noise)."""
        job = self.ideal_estimator.run([(ansatz, hamiltonian, np.asarray(theta, float))])
        return float(job.result()[0].data.evs)

    def calculate_noisy_energy(
        self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, theta: np.ndarray
    ) -> float:
        """Noisy energy (dengan noise model)."""
        job = self.noisy_estimator.run([(ansatz, hamiltonian, np.asarray(theta, float))])
        return float(job.result()[0].data.evs)

    def calculate_zne_energy(
        self,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        theta: np.ndarray,
        noise_factors: List[int] = [1, 3],
        do_extrapolate: bool = True,
    ) -> float:
        """ZNE via digital gate-folding pada gate 2-qubit, lalu ekstrapolasi linear."""
        theta_arr = np.asarray(theta, float)
        energies, x_vals = [], []

        for factor in noise_factors:
            try:
                if factor % 2 == 0 or factor < 1:
                    raise ValueError(f"noise_factor harus ganjil positif, dapat {factor}")

                if factor == 1:
                    folded = ansatz.copy()
                else:
                    n_pairs = (factor - 1) // 2
                    folded = QuantumCircuit(*ansatz.qregs, *ansatz.cregs)
                    for inst in ansatz.data:
                        op, qubits, clbits = inst.operation, inst.qubits, inst.clbits
                        folded.append(op, qubits, clbits)
                        if len(qubits) == 2 and op.name not in ("barrier", "measure"):
                            op_inv = op.inverse()
                            for _ in range(n_pairs):
                                folded.append(op_inv, qubits, clbits)
                                folded.append(op, qubits, clbits)

                job = self.noisy_estimator.run([(folded, hamiltonian, theta_arr)])
                energies.append(float(job.result()[0].data.evs))
                x_vals.append(factor)

            except Exception as e:
                logger.warning(f"ZNE failed for factor {factor}: {e}")
                energies.append(np.nan)
                x_vals.append(factor)

        energies_arr, x_arr = np.array(energies, float), np.array(x_vals, float)

        if not do_extrapolate:
            return energies_arr  # type: ignore[return-value]

        mask = np.isfinite(energies_arr)
        x_fit, y_fit = x_arr[mask], energies_arr[mask]
        if len(x_fit) < 2:
            logger.warning("Data tidak cukup untuk ekstrapolasi – mengembalikan rata-rata.")
            return float(np.mean(y_fit))

        _, c = np.polyfit(x_fit, y_fit, deg=1)
        return float(c)

    def calculate_all_energies(
        self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, theta: np.ndarray
    ) -> EnergyResult:
        """Hitung ideal (exact), noisy, dan ZNE energi sekaligus."""
        return EnergyResult(
            ideal_energy=self.calculate_groundstate(hamiltonian),
            noisy_energy=self.calculate_noisy_energy(ansatz, hamiltonian, theta),
            zne_energy=self.calculate_zne_energy(ansatz, hamiltonian, theta),
            theta_opt=theta,
        )


# ============================================================================
# 3. ML MODEL TRAINING
# ============================================================================

class TrainML:
    """Latih RandomForest model untuk ML-based error mitigation."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        noisy_energies: np.ndarray,
        ideal_energies: np.ndarray,
        observables: List[str],
        parameters: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build feature matrix para training RF model."""
        n_qubits = len(observables[0]) if observables else 4
        rows = []

        for noisy_e, obs, params in zip(noisy_energies, observables, parameters):
            feat: Dict = {"noisy_energy": float(noisy_e)}
            for i, p in enumerate(params):
                feat[f"param_{i}"] = float(p)
            # One-hot observable
            for pos in range(n_qubits):
                for op in "IXYZ":
                    feat[f"obs{pos}{op}"] = 0
            for pos, ch in enumerate(obs):
                feat[f"obs{pos}{ch}"] = 1
            rows.append(feat)

        X = pd.DataFrame(rows)
        self.feature_names = list(X.columns)
        return X, np.asarray(ideal_energies)

    # ------------------------------------------------------------------
    # Train / predict / persist
    # ------------------------------------------------------------------

    def train_model(
        self, X: pd.DataFrame, y: np.ndarray, n_estimators: int = 100, **kwargs
    ) -> Dict:
        """Train RandomForest model dan kembalikan metrik evaluasi."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=self.random_state, **kwargs
        )
        self.model.fit(X_train, y_train)

        metrics = {
            "train_mse": mean_squared_error(y_train, self.model.predict(X_train)),
            "test_mse": mean_squared_error(y_test, self.model.predict(X_test)),
            "train_r2": r2_score(y_train, self.model.predict(X_train)),
            "test_r2": r2_score(y_test, self.model.predict(X_test)),
            "n_samples": len(X),
            "n_features": X.shape[1],
        }
        logger.info(f"✓ RF model trained: Test MSE={metrics['test_mse']:.6f}, R²={metrics['test_r2']:.4f}")
        return metrics

    def save_model(self, path: str):
        if self.model is None:
            logger.warning("Model not trained. Cannot save.")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"✓ Model saved to {path}")

    def load_model(self, path: str):
        self.model = joblib.load(path)
        self.feature_names = list(self.model.feature_names_in_)
        logger.info(f"✓ Model loaded from {path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained/loaded")
        return self.model.predict(X)


# ============================================================================
# 4. VQE RUNNER
# ============================================================================

class VQERunner:
    """Jalankan VQE dengan ideal, noisy, ZNE, dan ML-mitigated backends."""

    def __init__(
        self,
        n_qubits: int = 4,
        maxiter: int = 125,
        seed: int = 170,
        ml_model_path: Optional[str] = None,
    ):
        self.n_qubits = n_qubits
        self.maxiter = maxiter
        self.seed = seed
        algorithm_globals.random_seed = seed

        self.energy_calc = CalculateEnergy(n_qubits)
        self.ml_trainer = TrainML()
        if ml_model_path:
            self.ml_trainer.load_model(ml_model_path)

        self.history: Dict = {
            "ideal": {"value": None},
            "noisy": {"counts": [], "values": []},
            "zne": {"counts": [], "values": []},
            "mitigated": {"counts": [], "values": []},
        }

    # ------------------------------------------------------------------
    # Individual runners
    # ------------------------------------------------------------------

    def run_vqe_ideal(self, hamiltonian: SparsePauliOp) -> Dict:
        """Exact ground state via NumPyMinimumEigensolver."""
        logger.info("=" * 70)
        logger.info("VQE IDEAL (Exact - NumPyMinimumEigensolver)")
        logger.info("=" * 70)
        exact_energy = self.energy_calc.calculate_groundstate(hamiltonian)
        self.history["ideal"]["value"] = exact_energy
        return {"energy": exact_energy, "method": "exact_diagonalization", "history": {"value": exact_energy}}

    def _run_vqe_spsa(
        self, label: str, cost_fn_builder, ansatz: QuantumCircuit
    ) -> Dict:
        """Generic SPSA runner – dipakai oleh noisy, zne, dan mitigated."""
        history: Dict = {"counts": [], "values": []}

        def cost_fn(theta):
            count = len(history["counts"]) + 1
            energy = cost_fn_builder(theta)
            history["counts"].append(count)
            history["values"].append(energy)
            if count % 10 == 0:
                logger.info(f" {label} Iter {count}: E = {energy:.6f}")
            return energy

        algorithm_globals.random_seed = self.seed
        x0 = 2 * np.pi * np.random.random(ansatz.num_parameters)
        result = SPSA(maxiter=self.maxiter).minimize(cost_fn, x0=x0)
        logger.info(f"✓ {label} VQE completed. Final energy: {result.fun:.6f}")
        self.history[label.lower()] = history
        return {"energy": result.fun, "theta_opt": result.x, "history": history}

    def run_vqe_noisy(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp) -> Dict:
        logger.info("=" * 70)
        logger.info("VQE NOISY (With Noise - SPSA Optimization)")
        logger.info("=" * 70)
        return self._run_vqe_spsa(
            "Noisy",
            lambda theta: self.energy_calc.calculate_noisy_energy(ansatz, hamiltonian, theta),
            ansatz,
        )

    def run_vqe_zne(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp) -> Dict:
        logger.info("=" * 70)
        logger.info("VQE ZNE (Zero Noise Extrapolation - SPSA Optimization)")
        logger.info("=" * 70)
        return self._run_vqe_spsa(
            "ZNE",
            lambda theta: self.energy_calc.calculate_zne_energy(ansatz, hamiltonian, theta),
            ansatz,
        )

    def run_vqe_mitigated(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp) -> Dict:
        if self.ml_trainer.model is None:
            logger.warning("ML model not loaded. Skipping mitigated VQE.")
            return {"status": "skipped"}
        logger.info("=" * 70)
        logger.info("VQE ML-MITIGATED (Noisy + RF Correction)")
        logger.info("=" * 70)
        return self._run_vqe_spsa(
            "Mitigated",
            lambda theta: self._compute_mitigated_energy(ansatz, hamiltonian, theta),
            ansatz,
        )

    def _compute_mitigated_energy(
        self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, theta: np.ndarray
    ) -> float:
        """Compute ML-corrected energy term-by-term."""
        theta_arr = np.asarray(theta, float)
        H_op = hamiltonian if isinstance(hamiltonian, SparsePauliOp) else SparsePauliOp(hamiltonian)
        labels = H_op.paulis.to_labels()
        coeffs = np.real(H_op.coeffs)

        pubs, active_labels, active_coeffs = [], [], []
        identity_shift = 0.0

        for label, coeff in zip(labels, coeffs):
            if np.isclose(coeff, 0.0):
                continue
            if set(label) == {"I"}:
                identity_shift = float(coeff)
                continue
            pubs.append((ansatz, SparsePauliOp(label), theta_arr))
            active_labels.append(label)
            active_coeffs.append(float(coeff))

        if not pubs:
            return identity_shift

        results = self.energy_calc.noisy_estimator.run(pubs).result()
        rows = []
        for res, label in zip(results, active_labels):
            noisy_exp = float(np.asarray(res.data.evs).ravel()[0])
            feat: Dict = {"noisy_energy": noisy_exp}
            for i, p in enumerate(theta_arr):
                feat[f"param_{i}"] = float(p)
            for pos in range(len(label)):
                for op in "IXYZ":
                    feat[f"obs{pos}{op}"] = 0
            for pos, ch in enumerate(label):
                feat[f"obs{pos}{ch}"] = 1
            rows.append(feat)

        X_df = pd.DataFrame(rows, columns=self.ml_trainer.feature_names)
        pred_terms = self.ml_trainer.predict(X_df)
        return identity_shift + float(np.dot(active_coeffs, pred_terms))

    def run_all(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp) -> Dict:
        """Jalankan semua VQE scenario (ideal, noisy, zne, mitigated)."""
        return {
            "ideal": self.run_vqe_ideal(hamiltonian),
            "noisy": self.run_vqe_noisy(ansatz, hamiltonian),
            "zne": self.run_vqe_zne(ansatz, hamiltonian),
            "mitigated": self.run_vqe_mitigated(ansatz, hamiltonian),
        }


# ============================================================================
# 5. EXTENDED PIPELINE – Training & Model Management
# ============================================================================

class ExtendedPipeline:
    """
    Extended Pipeline untuk:
    - Generate training dataset dengan multiple observables
    - Train ML model (RandomForest)
    - Evaluate model pada test data
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_circuits: int = 2000,
        output_dir: str = "output",
        noise_model: Optional[NoiseModel] = None,
        ansatz_config: Optional[AnsatzConfig] = None,   # <-- BARU
        theta_range: Tuple[float, float] = (-np.pi, np.pi),  # <-- BARU
    ):
        self.n_qubits = n_qubits
        self.n_circuits = n_circuits
        self.output_dir = output_dir
        self.noise_model = noise_model
        self.theta_range = theta_range

        for sub in ("train_data", "ml_model", "predict_data"):
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

        # Gunakan ansatz_config dari luar jika ada, fallback ke default
        config = ansatz_config if ansatz_config is not None else AnsatzConfig(n_qubits=n_qubits)
        self.ansatz_gen = GenerateAnsatz(config)
        self.energy_calc = CalculateEnergy(n_qubits, noise_model=noise_model)
        self.ml_trainer = TrainML()
        logger.info(f"ExtendedPipeline initialized: {n_circuits} circuits, {n_qubits} qubits")

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_training_dataset(
        self, observable_list: List[str], model_id: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Generate training dataset dengan multiple observables."""
        logger.info(f"Generating dataset: {self.n_circuits} circuits, {len(observable_list)} observables")
        base_ansatz = self.ansatz_gen.create_ansatz()
        n_obs = len(observable_list)

        noisy_list, ideal_list, zne_list, obs_list, param_list = [], [], [], [], []

        for idx in range(self.n_circuits):
            pauli = observable_list[idx % n_obs]
            obs_op = SparsePauliOp(pauli)
            theta = np.random.uniform(self.theta_range[0], self.theta_range[1], size=base_ansatz.num_parameters)

            ideal_list.append(float(self.energy_calc.calculate_ideal_energy(base_ansatz, obs_op, theta)))
            noisy_list.append(float(self.energy_calc.calculate_noisy_energy(base_ansatz, obs_op, theta)))
            zne_list.append(float(self.energy_calc.calculate_zne_energy(base_ansatz, obs_op, theta)))
            obs_list.append(pauli)
            param_list.append(theta)

            if (idx + 1) % 100 == 0:
                logger.info(f"{idx+1}/{self.n_circuits} circuits processed")

        noisy_arr = np.array(noisy_list, float)
        ideal_arr = np.array(ideal_list, float)
        zne_arr = np.array(zne_list, float)
        param_arr = np.array(param_list, float)

        self._save_dataset_csv(noisy_arr, ideal_arr, zne_arr, obs_list, param_arr, model_id)
        logger.info("✓ Dataset generation completed")
        return noisy_arr, ideal_arr, obs_list, param_arr

    def _save_dataset_csv(
        self,
        noisy_energies: np.ndarray,
        ideal_energies: np.ndarray,
        zne_energies: np.ndarray,
        observables: List[str],
        parameters: np.ndarray,
        model_id: int,
    ):
        rows = []
        for i in range(len(noisy_energies)):
            row = {
                "noisy_energy": noisy_energies[i],
                "ideal_energy": ideal_energies[i],
                "zne_energy": zne_energies[i],
                "observable": observables[i],
            }
            row.update({f"param_{j}": p for j, p in enumerate(parameters[i])})
            rows.append(row)

        csv_path = os.path.join(self.output_dir, "train_data", f"train_data_{model_id}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"✓ Dataset saved to: {csv_path}")

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train_model(
        self,
        noisy_energies: np.ndarray,
        ideal_energies: np.ndarray,
        observables: List[str],
        parameters: np.ndarray,
        model_id: int = 1,
        n_estimators: int = 100,
    ) -> Dict:
        """Train ML model (RandomForest) dan simpan ke disk."""
        logger.info("Training ML model...")
        X, y = self.ml_trainer.build_dataset(noisy_energies, ideal_energies, observables, parameters)
        metrics = self.ml_trainer.train_model(X, y, n_estimators=n_estimators)

        model_path = os.path.join(self.output_dir, "ml_model", f"qem_model_{model_id}.pkl")
        self.ml_trainer.save_model(model_path)
        logger.info(f"✓ Model trained and saved to: {model_path}")
        return metrics

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def evaluate_model(self, test_data_path: str, model_id: int = 1) -> pd.DataFrame:
        """Evaluate model pada test data CSV, simpan prediksi."""
        logger.info(f"Evaluating model on: {test_data_path}")
        test_data = pd.read_csv(test_data_path)

        noisy_energies = np.array(test_data["noisy_energy"], float)
        ideal_energies = np.array(test_data["ideal_energy"], float)
        observables = test_data["observable"].astype(str).tolist()

        if self.ansatz_gen.ansatz is None:
            self.ansatz_gen.create_ansatz()
        n_params = self.ansatz_gen.ansatz.num_parameters
        parameters = test_data[[f"param_{i}" for i in range(n_params)]].to_numpy()

        X_test, _ = self.ml_trainer.build_dataset(noisy_energies, ideal_energies, observables, parameters)
        test_data["RF_energy"] = self.ml_trainer.predict(X_test)

        pred_path = os.path.join(self.output_dir, "predict_data", f"predicted_data_{model_id}.csv")
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        test_data.to_csv(pred_path, index=False)
        logger.info(f"✓ Predictions saved to: {pred_path}")
        return test_data


# ============================================================================
# 6. VQE PIPELINE – Single & Multiple Bond Lengths
# ============================================================================

class VQEPipeline:
    """
    Orchestrate full ML-QEM VQE pipeline:
    - Single bond length VQE
    - Multiple bond lengths VQE
    """

    def __init__(
        self,
        n_qubits: int = 4,
        output_dir: str = "qem_output",
        noise_model: Optional[NoiseModel] = None,
        ansatz_config: Optional[AnsatzConfig] = None,   # <-- BARU
    ):
        self.n_qubits = n_qubits
        self.output_dir = output_dir
        self.noise_model = noise_model
        os.makedirs(output_dir, exist_ok=True)

        # Gunakan ansatz_config dari luar jika ada, fallback ke default
        config = ansatz_config if ansatz_config is not None else AnsatzConfig(n_qubits=n_qubits)
        self.ansatz_gen = GenerateAnsatz(config)
        self.energy_calc = CalculateEnergy(n_qubits, noise_model=noise_model)
        self.ml_trainer = TrainML()
        self.h2_gen = H2HamiltonianGenerator(n_qubits=n_qubits)
        logger.info(f"VQEPipeline initialized: {n_qubits} qubits")

    # ------------------------------------------------------------------
    # Single bond length
    # ------------------------------------------------------------------

    def run_full_vqe_pipeline(
        self,
        hamiltonian: SparsePauliOp,
        ml_model_path: Optional[str] = None,
        maxiter: int = 125,
    ) -> Dict:
        """Run full VQE pipeline untuk SATU Hamiltonian."""
        logger.info("=" * 70)
        logger.info("QEM FULL VQE PIPELINE (SINGLE BOND LENGTH)")
        logger.info("=" * 70)

        ansatz = self.ansatz_gen.create_ansatz()
        logger.info(f"Ansatz info: {self.ansatz_gen.get_info()}")

        results = VQERunner(
            n_qubits=self.n_qubits, maxiter=maxiter, ml_model_path=ml_model_path
        ).run_all(ansatz, hamiltonian)

        self._save_results(results)
        return results

    def _save_results(self, results: Dict):
        """Save results single bond length ke CSV."""
        rows = [
            {
                "key": key,
                "status": res.get("status", "done"),
                "energy": res.get("energy"),
                "method": res.get("method", "optimization"),
            }
            for key, res in results.items()
        ]
        csv_path = os.path.join(self.output_dir, "vqe_results.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"✓ Results saved to {csv_path}")

    # ------------------------------------------------------------------
    # Multiple bond lengths
    # ------------------------------------------------------------------

    def run_full_vqe_pipeline_multi_bond_lengths(
        self,
        bond_lengths: List[float],
        ml_model_path: Optional[str] = None,
        maxiter: int = 125,
        save_results: bool = True,
    ) -> Dict[float, Dict]:
        """Run full VQE pipeline untuk MULTIPLE bond lengths."""
        logger.info("=" * 70)
        logger.info("QEM FULL VQE PIPELINE (MULTIPLE BOND LENGTHS)")
        logger.info("=" * 70)
        logger.info(f"Bond lengths: {bond_lengths}")

        hamiltonians = self.h2_gen.generate_hamiltonian_range(np.array(bond_lengths, float))
        ansatz = self.ansatz_gen.create_ansatz()
        logger.info(f"✓ Ansatz created: {self.ansatz_gen.get_info()}")

        vqe_runner = VQERunner(
            n_qubits=self.n_qubits, maxiter=maxiter, ml_model_path=ml_model_path
        )

        results_all: Dict[float, Dict] = {}
        total = len(hamiltonians)

        for idx, (bond_length, hamiltonian) in enumerate(hamiltonians.items(), 1):
            logger.info(f"{'='*70}")
            logger.info(f"[{idx}/{total}] Bond Length: {bond_length:.4f} Å")
            logger.info(f"{'='*70}")
            try:
                res = vqe_runner.run_all(ansatz, hamiltonian)
                results_all[bond_length] = res
                logger.info(f"✓ Bond length {bond_length:.4f} completed:")
                logger.info(f"  - Ideal:    {res['ideal'].get('energy', 'N/A'):.6f}")
                logger.info(f"  - Noisy:    {res['noisy'].get('energy', 'N/A'):.6f}")
                logger.info(f"  - ZNE:      {res['zne'].get('energy', 'N/A'):.6f}")
                if res["mitigated"].get("status") != "skipped":
                    logger.info(f"  - Mitigated: {res['mitigated'].get('energy', 'N/A'):.6f}")
            except Exception as e:
                logger.error(f"✗ Error at bond_length {bond_length:.4f}: {e}")
                results_all[bond_length] = {"status": "error", "error": str(e)}

        if save_results:
            self._save_multi_bond_length_results(results_all)

        # Summary table
        summary_rows = []
        for bl in sorted(results_all.keys()):
            res = results_all[bl]
            if res.get("status") == "error":
                summary_rows.append({"bond_length": f"{bl:.4f}", "ideal": "ERROR",
                                     "noisy": "ERROR", "zne": "ERROR", "mitigated": "ERROR"})
            else:
                mit_e = (f"{res['mitigated'].get('energy', np.nan):.6f}"
                         if res["mitigated"].get("status") != "skipped" else "SKIPPED")
                summary_rows.append({
                    "bond_length": f"{bl:.4f}",
                    "ideal": f"{res['ideal'].get('energy', np.nan):.6f}",
                    "noisy": f"{res['noisy'].get('energy', np.nan):.6f}",
                    "zne": f"{res['zne'].get('energy', np.nan):.6f}",
                    "mitigated": mit_e,
                })
        logger.info("\n" + pd.DataFrame(summary_rows).to_string(index=False))
        return results_all

    def _save_multi_bond_length_results(self, results_all: Dict[float, Dict]):
        rows = []
        for bl in sorted(results_all.keys()):
            res = results_all[bl]
            if res.get("status") == "error":
                rows.append({"bond_length": float(bl), "ideal_energy": np.nan,
                             "noisy_energy": np.nan, "zne_energy": np.nan,
                             "mitigated_energy": np.nan, "status": "error",
                             "error_msg": res.get("error")})
            else:
                mit_e = (np.nan if res["mitigated"].get("status") == "skipped"
                         else res["mitigated"].get("energy", np.nan))
                rows.append({
                    "bond_length": float(bl),
                    "ideal_energy": res["ideal"].get("energy", np.nan),
                    "noisy_energy": res["noisy"].get("energy", np.nan),
                    "zne_energy": res["zne"].get("energy", np.nan),
                    "mitigated_energy": mit_e,
                    "status": "done" if res["mitigated"].get("status") != "skipped" else "mitigated_skipped",
                    "error_msg": None,
                })

        csv_path = os.path.join(self.output_dir, "vqe_results_multi_bond_length.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"✓ Multi-bond-length results saved to: {csv_path}")


# ============================================================================
# 7. UNIFIED PIPELINE – End-to-End Workflow dengan Experiment ID
# ============================================================================

class UnifiedPipeline:
    """
    Complete end-to-end pipeline:
      Phase 1 – ML Training:   Generate dataset + train model
      Phase 2 – ML Evaluation: Test model accuracy
      Phase 3 – VQE Execution: Use trained model untuk error mitigation

    Output diorganisir dalam folder experiment_id; filename tetap standar.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_circuits: int = 2000,
        output_dir: str = "unified_output",
        experiment_id: Optional[str] = None,
        noise_model: Optional[NoiseModel] = None,
        ansatz_config: Optional[AnsatzConfig] = None,          # <-- BARU
        theta_range: Tuple[float, float] = (-np.pi, np.pi),    # <-- BARU
    ):
        self.n_qubits = n_qubits
        self.n_circuits = n_circuits
        self.output_dir = output_dir
        self.noise_model = noise_model

        self.experiment_id = experiment_id or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir = os.path.join(output_dir, self.experiment_id)
        self.ml_dir = os.path.join(self.exp_dir, "ml_training")
        self.eval_dir = os.path.join(self.exp_dir, "ml_evaluation")
        self.vqe_dir = os.path.join(self.exp_dir, "vqe_execution")

        for d in (self.ml_dir, self.eval_dir, self.vqe_dir):
            os.makedirs(d, exist_ok=True)

        # Teruskan ansatz_config dan theta_range ke kedua sub-pipeline
        self.extended_pipeline = ExtendedPipeline(
            n_qubits=n_qubits, n_circuits=n_circuits,
            output_dir=self.ml_dir, noise_model=noise_model,
            ansatz_config=ansatz_config,
            theta_range=theta_range,
        )
        self.vqe_pipeline = VQEPipeline(
            n_qubits=n_qubits, output_dir=self.vqe_dir, noise_model=noise_model,
            ansatz_config=ansatz_config,
        )

        self.model_path: Optional[str] = None
        self.model_metrics: Optional[Dict] = None

        logger.info(f"UnifiedPipeline initialized – Experiment ID: {self.experiment_id}")
        logger.info(f"Output directory: {self.exp_dir}")

    # ------------------------------------------------------------------
    # Phase 1: ML Training
    # ------------------------------------------------------------------

    def phase1_ml_training(
        self,
        observable_list: List[str],
        model_id: int = 1,
        n_estimators: int = 100,
    ) -> Dict:
        """Generate dataset dan train ML model."""
        logger.info("=" * 70)
        logger.info(f"PHASE 1: ML TRAINING  [Experiment: {self.experiment_id}]")
        logger.info("=" * 70)

        noisy, ideal, observables, parameters = \
            self.extended_pipeline.generate_training_dataset(
                observable_list=observable_list, model_id=model_id
            )

        metrics = self.extended_pipeline.train_model(
            noisy, ideal, observables, parameters,
            model_id=model_id, n_estimators=n_estimators,
        )

        self.model_path = os.path.join(self.ml_dir, "ml_model", f"qem_model_{model_id}.pkl")
        self.model_metrics = metrics

        logger.info("✓ PHASE 1 COMPLETED")
        return {
            "status": "success",
            "experiment_id": self.experiment_id,
            "dataset_path": os.path.join(self.ml_dir, "train_data", f"train_data_{model_id}.csv"),
            "model_path": self.model_path,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # Phase 2: ML Evaluation
    # ------------------------------------------------------------------

    def phase2_ml_evaluation(self, test_data_path: str, model_id: int = 1) -> Dict:
        """Evaluate trained model pada test data."""
        if self.model_path is None:
            raise ValueError("Model path not set. Run phase1_ml_training() first.")

        logger.info("=" * 70)
        logger.info(f"PHASE 2: ML EVALUATION  [Experiment: {self.experiment_id}]")
        logger.info("=" * 70)

        self.extended_pipeline.ml_trainer.load_model(self.model_path)
        test_results = self.extended_pipeline.evaluate_model(
            test_data_path=test_data_path, model_id=model_id
        )

        eval_metrics: Dict = {}
        if {"ideal_energy", "RF_energy"}.issubset(test_results.columns):
            y_true = test_results["ideal_energy"]
            y_pred = test_results["RF_energy"]
            eval_metrics = {
                "mse": mean_squared_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
            }
            for k, v in eval_metrics.items():
                logger.info(f"  - {k.upper()}: {v:.6f}")

        pred_path = os.path.join(self.eval_dir, f"predictions_{model_id}.csv")
        test_results.to_csv(pred_path, index=False)
        logger.info(f"✓ Predictions saved: {pred_path}")
        logger.info("✓ PHASE 2 COMPLETED")

        return {
            "status": "success",
            "experiment_id": self.experiment_id,
            "predictions_path": pred_path,
            "metrics": eval_metrics,
            "n_samples": len(test_results),
        }

    # ------------------------------------------------------------------
    # Phase 3a: VQE – Single Bond Length
    # ------------------------------------------------------------------

    def phase3_vqe_single_bond(self, bond_length: float, maxiter: int = 125) -> Dict:
        """VQE untuk single bond length dengan ML mitigation."""
        if self.model_path is None:
            raise ValueError("Model path not set. Run phase1_ml_training() first.")

        logger.info("=" * 70)
        logger.info(f"PHASE 3a: VQE (Bond length: {bond_length} Å)  [Experiment: {self.experiment_id}]")
        logger.info("=" * 70)

        hamiltonian = self.vqe_pipeline.h2_gen.generate_h2_hamiltonian(bond_length)
        results = self.vqe_pipeline.run_full_vqe_pipeline(
            hamiltonian=hamiltonian, ml_model_path=self.model_path, maxiter=maxiter
        )
        logger.info("✓ PHASE 3a COMPLETED")
        return results

    # ------------------------------------------------------------------
    # Phase 3b: VQE – Multiple Bond Lengths
    # ------------------------------------------------------------------

    def phase3_vqe_multi_bonds(
        self, bond_lengths: List[float], maxiter: int = 125
    ) -> Dict[float, Dict]:
        """VQE untuk multiple bond lengths dengan ML mitigation."""
        if self.model_path is None:
            raise ValueError("Model path not set. Run phase1_ml_training() first.")

        logger.info("=" * 70)
        logger.info(f"PHASE 3b: VQE Multi-Bond  [Experiment: {self.experiment_id}]")
        logger.info("=" * 70)

        results = self.vqe_pipeline.run_full_vqe_pipeline_multi_bond_lengths(
            bond_lengths=bond_lengths, ml_model_path=self.model_path,
            maxiter=maxiter, save_results=True,
        )
        logger.info("✓ PHASE 3b COMPLETED")
        return results

    # ------------------------------------------------------------------
    # Complete workflow
    # ------------------------------------------------------------------

    def run_complete_workflow(
        self,
        observable_list: List[str],
        test_data_path: str,
        bond_lengths: List[float],
        model_id: int = 1,
        n_estimators: int = 100,
        maxiter: int = 125,
    ) -> Dict:
        """Jalankan Phase 1 → 2 → 3 secara berurutan."""
        logger.info("\n" + "=" * 70)
        logger.info(f"🚀 UNIFIED PIPELINE – COMPLETE WORKFLOW  [ID: {self.experiment_id}]")
        logger.info("=" * 70)

        workflow: Dict = {}

        for phase_name, phase_fn, phase_kwargs in [
            ("phase1_training",  self.phase1_ml_training,
             {"observable_list": observable_list, "model_id": model_id, "n_estimators": n_estimators}),
            ("phase2_evaluation", self.phase2_ml_evaluation,
             {"test_data_path": test_data_path, "model_id": model_id}),
            ("phase3_vqe",       self.phase3_vqe_multi_bonds,
             {"bond_lengths": bond_lengths, "maxiter": maxiter}),
        ]:
            try:
                workflow[phase_name] = phase_fn(**phase_kwargs)  # type: ignore[operator]
            except Exception as e:
                logger.error(f"{phase_name} failed: {e}")
                return {"status": "error", "phase": phase_name, "error": str(e)}

        logger.info("\n" + "=" * 70)
        logger.info(f"✅ WORKFLOW COMPLETE  [ID: {self.experiment_id}]")
        logger.info(f"Output: {self.exp_dir}/")
        logger.info("=" * 70)
        return workflow

    def get_workflow_summary(self) -> str:
        return (
            f"\n{'='*76}\n"
            f"  UNIFIED PIPELINE WORKFLOW SUMMARY\n"
            f"  Experiment ID : {self.experiment_id}\n"
            f"  Output Dir    : {self.exp_dir}\n"
            f"{'='*76}\n"
            "PHASE 1 – ML Training   : generate dataset, train RandomForest\n"
            "PHASE 2 – ML Evaluation : test accuracy (MSE, R², MAE)\n"
            "PHASE 3 – VQE Execution : ideal | noisy | ZNE | ML-mitigated\n"
        )


# ============================================================================
# 8. UTILITY FUNCTION
# ============================================================================

def create_h2_hamiltonian(bond_length: float, n_qubits: int = 4) -> SparsePauliOp:
    """Quick utility untuk generate H2 Hamiltonian."""
    return H2HamiltonianGenerator(n_qubits=n_qubits).generate_h2_hamiltonian(bond_length)


# ============================================================================
# 9. MAIN (CLI entry point)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ML-QEM Pipeline")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=125)
    parser.add_argument("--seed", type=int, default=170)
    parser.add_argument("--bond-length", type=float, default=0.735)
    parser.add_argument("--model_path", type=str, default="output/ml_model/qem_model_1.pkl")
    parser.add_argument("--output-dir", type=str, default="output/vqe_data/vqe_data.csv")
    args = parser.parse_args()

    hamiltonian = H2HamiltonianGenerator(n_qubits=args.n_qubits).generate_h2_hamiltonian(args.bond_length)
    results = VQEPipeline(n_qubits=args.n_qubits, output_dir=args.output_dir).run_full_vqe_pipeline(
        hamiltonian=hamiltonian, maxiter=args.maxiter, ml_model_path=args.model_path
    )

    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    for key, res in results.items():
        if res.get("status") == "skipped":
            logger.info(f"{key.upper()}: SKIPPED")
        else:
            logger.info(f"{key.upper()}: E = {res.get('energy'):.6f}")
    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
