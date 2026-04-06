"""
qem_main_updated_v2.py - ML-QEM Pipeline dengan ZNE, H2 Hamiltonian, dan Extended Features

Fitur Baru:
1. ZNE Energy Calculation dengan ZneOptions
2. Extended Pipeline Workflow (seperti main.py)
3. H2 Hamiltonian Generator dengan variable bond length
4. VQE Runner dengan ZNE calculation
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict

from qiskit.circuit.library import n_local
from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import Layout
from qiskit_aer.primitives import EstimatorV2 as Estimator

from typing import List, Optional
from qiskit import QuantumCircuit

from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit_ibm_runtime import EstimatorV2 as ZNE_estimator
from custom_noise import CustomNoiseBuilder

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevMapper

from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import VQE, NumPyMinimumEigensolver

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
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
        """
        Generate H2 Hamiltonian dengan bond length tertentu.
        
        Parameters
        ----------
        bond_length : float
            Bond length H-H (Angstrom)
        
        Returns
        -------
        SparsePauliOp
            Hamiltonian untuk H2
        """
        atom_string = f"H 0 0 0; H 0 0 {bond_length}"

        driver = PySCFDriver(
            atom=atom_string,
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        
        es_problem = driver.run()
        second_q_op = es_problem.hamiltonian.second_q_op()

    # dapatkan qubit hammiltonian
        if self.n_qubits == 4:
            mapper = JordanWignerMapper()
        if self.n_qubits == 2:
            mapper = ParityMapper(num_particles=es_problem.num_particles)
        hamiltonian = mapper.map(second_q_op)
        return hamiltonian
    def generate_hamiltonian_range(
        self, 
        bond_lengths: np.ndarray
    ) -> Dict[float, SparsePauliOp]:
        """
        Generate Hamiltonian untuk range bond lengths.
        
        Parameters
        ----------
        bond_lengths : np.ndarray
            Array of bond lengths
        
        Returns
        -------
        Dict[float, SparsePauliOp]
            Dictionary mapping bond_length -> Hamiltonian
        """
        hamiltonians = {}
        for r in bond_lengths:
            hamiltonians[float(r)] = self.generate_h2_hamiltonian(r)
        
        logger.info(f"✓ Generated {len(hamiltonians)} Hamiltonians for bond lengths {bond_lengths}")
        return hamiltonians


# ============================================================================
# 1. ANSATZ GENERATION
# ============================================================================

@dataclass
class AnsatzConfig:
    """Configuration untuk ansatz generation."""
    n_qubits: int = 4
    entanglement_blocks: str = "cx"
    entanglement: str = "linear"
    reps: int = 2
    insert_barriers: bool = True
    param_range: Tuple[float, float] = (0, 2*np.pi)

class GenerateAnsatz:
    """Membuat quantum circuits dengan parameterized ansatz."""
    
    def __init__(self, config: AnsatzConfig):
        self.config = config
        self.ansatz = None
        self.n_circuits = 0
        logger.info(f"Ansatz config: {config.n_qubits} qubits, reps={config.reps}")
    
    def create_ansatz(self):
        """Buat ansatz circuit dengan NLocal."""
        self.ansatz = n_local(
            self.config.n_qubits,
            rotation_blocks=['ry'],
            entanglement_blocks=self.config.entanglement_blocks,
            entanglement=self.config.entanglement,
            reps=self.config.reps,
            insert_barriers=self.config.insert_barriers
        )
        
        n_params = self.ansatz.num_parameters
        logger.info(f"✓ Ansatz created: {n_params} parameters")
        return self.ansatz
    
    def get_info(self) -> Dict:
        """Get ansatz information."""
        if self.ansatz is None:
            self.create_ansatz()
        return {
            "n_qubits": self.config.n_qubits,
            "n_parameters": self.ansatz.num_parameters,
            "depth": self.ansatz.decompose().depth()
        }


# ============================================================================
# 2. ENERGY CALCULATION (dengan ZNE)
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
    """Hitung energi dengan multiple backends: ideal (exact), noisy, ZNE."""
    
    def __init__(self, n_qubits: int = 4, 
                 backend_class=FakeBelemV2,
                 noise_model: Optional[NoiseModel] = None  # <--- TAMBAHAN DI SINI
                 ):
        self.n_qubits = n_qubits
        self.backend_class = backend_class
        self.noise_model = noise_model
        
        # Ideal estimator
        self.ideal_estimator = Estimator(options={"default_precision": 1e-2})
        
        
        # Custom Noisy Estimator
        # builder = CustomNoiseBuilder()
        # noise_model = builder.custom_backend(target_qubits=[0])
        
        # Noisy estimator
        backend_fake = backend_class()
        # noise_model = NoiseModel.from_backend(backend_fake)
        
        self.noisy_estimator = Estimator(
            options={
                "default_precision": 1e-2,
                "backend_options": {"noise_model": noise_model}
            }
        )
        
        # ZNE estimator dengan ZneOptions
        
        self.backend_fake = backend_fake
        logger.info(f"Energy calculators initialized: ideal + noisy + ZNE")
    
    def calculate_groundstate(self, hamiltonian: SparsePauliOp) -> float:
        """
        Hitung ideal energy dengan exact diagonalization (NumPyMinimumEigensolver).
        """
        try:
            solver = NumPyMinimumEigensolver()
            result = solver.compute_minimum_eigenvalue(hamiltonian)
            energy = float(result.eigenvalue.real)
            logger.info(f"✓ Exact ground truth energy: {energy:.6f}")
            return energy
        except Exception as e:
            logger.error(f"Error computing exact energy: {e}")
            raise
    
    def calculate_ideal_energy(
        self,
        ansatz,
        hamiltonian: SparsePauliOp,
        theta: np.ndarray
    ) -> float:
        """Hitung ideal energy (tanpa noise)."""
        theta_arr = np.asarray(theta, dtype=float)
        pub = (ansatz, hamiltonian, theta_arr)
        job = self.ideal_estimator.run([pub])
        result = job.result()
        energy = float(result[0].data.evs)
        return energy
    
    def calculate_noisy_energy(
        self,
        ansatz,
        hamiltonian: SparsePauliOp,
        theta: np.ndarray
    ) -> float:
        """Hitung noisy energy (dengan noise model)."""
        theta_arr = np.asarray(theta, dtype=float)
        pub = (ansatz, hamiltonian, theta_arr)
        job = self.noisy_estimator.run([pub])
        result = job.result()
        energy = float(result[0].data.evs)
        return energy
    
    import numpy as np
    
    def calculate_zne_energy(
        self,
        ansatz,
        hamiltonian,
        theta: np.ndarray,
        noise_factors: List[float] = [1.0, 3.0, 5.0],
        do_extrapolate: bool = True,
    ):
        """
        Hitung ZNE (Zero Noise Extrapolation) energy.

        Mekanisme:
        - Amplifikasi noise dengan gate folding (menduplikasi circuit sesuai noise_factors).
        - Estimasi energi untuk tiap noise factor.
        - Ekstrapolasi ke noise=0 (default: linear 3 titik).
        """
        theta_arr = np.asarray(theta, dtype=float)

        energies: List[Optional[float]] = []
        x_vals: List[float] = []

        for factor in noise_factors:
            try:
                # --- 1) gate folding langsung di sini ---
                if factor == 1.0:
                    folded_ansatz = ansatz
                else:
                    k = int(factor)
                    if k % 2 == 0 or k < 1:
                        raise ValueError(
                            f"noise_factor harus ganjil positif, dapat {factor}"
                        )

                    folded_ansatz = QuantumCircuit(ansatz.num_qubits)
                    for _ in range(k):
                        folded_ansatz = folded_ansatz.compose(ansatz, inplace=False)

                # --- 2) jalankan estimator untuk circuit yang sudah di-fold ---
                pub = (folded_ansatz, hamiltonian, theta_arr)
                job = self.noisy_estimator.run([pub])
                result = job.result()
                energy = float(result[0].data.evs)

                energies.append(energy)
                x_vals.append(factor)

            except Exception as e:
                logger.warning(f"ZNE calculation failed for factor {factor}: {e}")
                energies.append(np.nan)
                x_vals.append(factor)

        energies_arr = np.array(energies, dtype=float)
        x_arr = np.array(x_vals, dtype=float)

        if not do_extrapolate:
            # kalau mau lihat semua energi per noise_factor
            return energies_arr  # atau list(energies_arr.tolist())

        # --- 3) ekstrapolasi linear 3 titik ke noise=0 di sini juga ---
        mask = np.isfinite(energies_arr)
        x_fit = x_arr[mask]
        y_fit = energies_arr[mask]

        if len(x_fit) < 2:
            # tidak cukup titik, pakai rata-rata
            return float(np.mean(y_fit))

        # y = a*x + b, ambil b di x=0
        coeffs = np.polyfit(x_fit, y_fit, deg=2)  # linear fit [web:33][web:43]
        a, b, c   = coeffs
        zne_energy = float(c)

        return zne_energy

    def calculate_all_energies(
        self,
        ansatz,
        hamiltonian: SparsePauliOp,
        theta: np.ndarray
    ) -> EnergyResult:
        """Hitung semua energi: ideal (exact), noisy, ZNE."""
        ideal = self.calculate_groundstate(hamiltonian)
        noisy = self.calculate_noisy_energy(ansatz, hamiltonian, theta)
        zne = self.calculate_zne_energy(ansatz, hamiltonian, theta)
        
        return EnergyResult(
            ideal_energy=ideal,
            noisy_energy=noisy,
            zne_energy=zne,
            theta_opt=theta
        )


# ============================================================================
# 3. ML MODEL TRAINING
# ============================================================================

class TrainML:
    """Latih RandomForest model untuk ML-based error mitigation."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.feature_names = None
    
    def build_dataset(
        self,
        noisy_energies: np.ndarray,
        ideal_energies: np.ndarray,
        observables: List[str],
        parameters: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build feature matrix untuk training RF model."""
        rows = []
        n_qubits = len(observables[0]) if observables else 4
        
        for noisy_e, obs, params in zip(noisy_energies, observables, parameters):
            feat = {}
            
            # 1. Noisy energy
            feat["noisy_energy"] = float(noisy_e)
            
            # 2. Parameters
            if parameters is not None:
                for i, p in enumerate(params):
                    feat[f"param_{i}"] = float(p)
            
            # 3. Observable one-hot
            for pos in range(n_qubits):
                for op in ["I", "X", "Y", "Z"]:
                    feat[f"obs{pos}{op}"] = 0
            
            for pos, ch in enumerate(obs):
                feat[f"obs{pos}{ch}"] = 1
            
            rows.append(feat)
        
        X = pd.DataFrame(rows)
        y = ideal_energies
        
        self.feature_names = list(X.columns)
        return X, y
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        n_estimators: int = 100,
        **kwargs
    ) -> Dict:
        """Train RandomForest model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state,
            **kwargs
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_mse = mean_squared_error(y_train, self.model.predict(X_train))
        test_mse = mean_squared_error(y_test, self.model.predict(X_test))
        train_r2 = r2_score(y_train, self.model.predict(X_train))
        test_r2 = r2_score(y_test, self.model.predict(X_test))
        
        metrics = {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "n_samples": len(X),
            "n_features": X.shape[1]
        }
        
        logger.info(f"✓ RF model trained: Test MSE={test_mse:.6f}, R²={test_r2:.4f}")
        return metrics
    
    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is None:
            logger.warning("Model not trained. Cannot save.")
            return
        
        joblib.dump(self.model, path)
        logger.info(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.feature_names = list(self.model.feature_names_in_)
        logger.info(f"✓ Model loaded from {path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained/loaded")
        
        logger.info("Predicting with ML model...")
        return self.model.predict(X)


# ============================================================================
# 4. VQE RUNNER (dengan ZNE dan ML mitigation)
# ============================================================================

class VQERunner:
    """Jalankan VQE dengan ideal (exact), noisy, ML-mitigated, dan ZNE backends."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        maxiter: int = 125,
        seed: int = 170,
        ml_model_path: Optional[str] = None
    ):
        self.n_qubits = n_qubits
        self.maxiter = maxiter
        self.seed = seed
        algorithm_globals.random_seed = seed
        
        # Energy calculator
        self.energy_calc = CalculateEnergy(n_qubits)
        
        # ML model
        self.ml_trainer = TrainML()
        if ml_model_path:
            self.ml_trainer.load_model(ml_model_path)
        
        self.history = {
            "ideal": {"value": None},
            "noisy": {"counts": [], "values": []},
            "zne": {"counts": [], "values": []},
            "mitigated": {"counts": [], "values": []}
        }
    
    def run_vqe_ideal(self, hamiltonian: SparsePauliOp) -> Dict:
        """Jalankan ideal VQE dengan exact diagonalization."""
        logger.info("=" * 70)
        logger.info("VQE IDEAL (Exact - NumPyMinimumEigensolver)")
        logger.info("=" * 70)
        
        try:
            exact_energy = self.energy_calc.calculate_groundstate(hamiltonian)
            self.history["ideal"]["value"] = exact_energy
            logger.info(f"✓ Ideal VQE completed (exact).")
            logger.info(f" Ground truth energy: {exact_energy:.6f}")
            
            return {
                "energy": exact_energy,
                "method": "exact_diagonalization",
                "history": {"value": exact_energy}
            }
        except Exception as e:
            logger.error(f"Error in ideal VQE: {e}")
            raise
    
    def run_vqe_noisy(self, ansatz, hamiltonian: SparsePauliOp) -> Dict:
        """Jalankan noisy VQE (tanpa mitigation)."""
        logger.info("=" * 70)
        logger.info("VQE NOISY (With Noise - SPSA Optimization)")
        logger.info("=" * 70)
        
        history = {"counts": [], "values": []}
        
        def cost_fn(theta):
            evalcount = len(history["counts"]) + 1
            energy = self.energy_calc.calculate_noisy_energy(
                ansatz, hamiltonian, theta
            )
            history["counts"].append(evalcount)
            history["values"].append(energy)
            
            if evalcount % 10 == 0:
                logger.info(f" Noisy Iter {evalcount}: E = {energy:.6f}")
            return energy
        
        num_params = ansatz.num_parameters
        algorithm_globals.random_seed = self.seed
        x0 = 2 * np.pi * np.random.random(num_params)
        
        optimizer = SPSA(maxiter=self.maxiter)
        logger.info(f"Starting optimization ({self.maxiter} iterations)...")
        result = optimizer.minimize(cost_fn, x0=x0)
        
        logger.info(f"✓ Noisy VQE completed.")
        logger.info(f" Final energy: {result.fun:.6f}")
        
        self.history["noisy"] = history
        return {
            "energy": result.fun,
            "theta_opt": result.x,
            "history": history
        }
    
    def run_vqe_zne(self, ansatz, hamiltonian: SparsePauliOp) -> Dict:
        """Jalankan ZNE-corrected VQE."""
        logger.info("=" * 70)
        logger.info("VQE ZNE (Zero Noise Extrapolation - SPSA Optimization)")
        logger.info("=" * 70)
        
        history = {"counts": [], "values": []}
        
        def cost_fn(theta):
            evalcount = len(history["counts"]) + 1
            energy = self.energy_calc.calculate_zne_energy(
                ansatz, hamiltonian, theta
            )
            history["counts"].append(evalcount)
            history["values"].append(energy)
            
            if evalcount % 10 == 0:
                logger.info(f" ZNE Iter {evalcount}: E = {energy:.6f}")
            return energy
        
        num_params = ansatz.num_parameters
        algorithm_globals.random_seed = self.seed
        x0 = 2 * np.pi * np.random.random(num_params)
        
        optimizer = SPSA(maxiter=self.maxiter)
        logger.info(f"Starting optimization ({self.maxiter} iterations)...")
        result = optimizer.minimize(cost_fn, x0=x0)
        
        logger.info(f"✓ ZNE VQE completed.")
        logger.info(f" Final energy: {result.fun:.6f}")
        
        self.history["zne"] = history
        return {
            "energy": result.fun,
            "theta_opt": result.x,
            "history": history
        }
    
    def run_vqe_mitigated(self, ansatz, hamiltonian: SparsePauliOp) -> Dict:
        """Jalankan ML-mitigated VQE."""
        if self.ml_trainer.model is None:
            logger.warning("ML model not loaded. Skipping mitigated VQE.")
            return {"status": "skipped"}
        
        logger.info("=" * 70)
        logger.info("VQE ML-MITIGATED (Noisy + RF Correction)")
        logger.info("=" * 70)
        
        history = {"counts": [], "values": []}
        
        def cost_fn(theta):
            evalcount = len(history["counts"]) + 1
            energy = self._compute_mitigated_energy(
                ansatz, hamiltonian, theta
            )
            history["counts"].append(evalcount)
            history["values"].append(energy)
            
            if evalcount % 10 == 0:
                logger.info(f" Mitigated Iter {evalcount}: E = {energy:.6f}")
            return energy
        
        num_params = ansatz.num_parameters
        algorithm_globals.random_seed = self.seed
        x0 = 2 * np.pi * np.random.random(num_params)
        
        optimizer = SPSA(maxiter=self.maxiter)
        logger.info(f"Starting optimization ({self.maxiter} iterations)...")
        result = optimizer.minimize(cost_fn, x0=x0)
        
        logger.info(f"✓ Mitigated VQE completed.")
        logger.info(f" Final energy: {result.fun:.6f}")
        
        self.history["mitigated"] = history
        return {
            "energy": result.fun,
            "theta_opt": result.x,
            "history": history
        }
    
    def _compute_mitigated_energy(
        self,
        ansatz,
        hamiltonian: SparsePauliOp,
        theta: np.ndarray
    ) -> float:
        """Compute ML-corrected energy."""
        theta_arr = np.asarray(theta, dtype=float)
        
        if not isinstance(hamiltonian, SparsePauliOp):
            H_op = SparsePauliOp(hamiltonian)
        else:
            H_op = hamiltonian
        
        labels = H_op.paulis.to_labels()
        coeffs = np.real(H_op.coeffs)
        
        pubs = []
        active_labels = []
        active_coeffs = []
        identity_shift = 0.0
        
        for label, coeff in zip(labels, coeffs):
            if np.isclose(coeff, 0.0):
                continue
            
            if set(label) == {"I"}:
                identity_shift = float(coeff)
                continue
            
            obs = SparsePauliOp(label)
            pubs.append((ansatz, obs, theta_arr))
            active_labels.append(label)
            active_coeffs.append(float(coeff))
        
        if not pubs:
            return identity_shift
        
        job = self.energy_calc.noisy_estimator.run(pubs)
        results = job.result()
        
        rows = []
        for res, label in zip(results, active_labels):
            vals = np.asarray(res.data.evs)
            noisy_exp = float(vals.ravel()[0])
            
            feat = {}
            feat["noisy_energy"] = noisy_exp
            
            for i, p in enumerate(theta_arr):
                feat[f"param_{i}"] = float(p)
            
            for pos in range(len(label)):
                for op in ["I", "X", "Y", "Z"]:
                    feat[f"obs{pos}{op}"] = 0
            
            for pos, ch in enumerate(label):
                feat[f"obs{pos}{ch}"] = 1
            
            rows.append(feat)
        
        X_df = pd.DataFrame(rows, columns=self.ml_trainer.feature_names)
        pred_terms = self.ml_trainer.predict(X_df)
        
        energy = identity_shift + float(np.dot(active_coeffs, pred_terms))
        return energy
    
    def run_all(self, ansatz, hamiltonian: SparsePauliOp) -> Dict:
        """Jalankan semua VQE scenario."""
        results = {}
        
        # Ideal: exact energy
        results["ideal"] = self.run_vqe_ideal(hamiltonian)
        
        # Noisy: SPSA dengan noise
        results["noisy"] = self.run_vqe_noisy(ansatz, hamiltonian)
        
        # ZNE: ZNE extrapolation
        results["zne"] = self.run_vqe_zne(ansatz, hamiltonian)
        
        # Mitigated: ML correction
        results["mitigated"] = self.run_vqe_mitigated(ansatz, hamiltonian)
        
        return results


# ============================================================================
# 5. EXTENDED WORKFLOW PIPELINE
# ============================================================================

class ExtendedPipeline:
    """
    Extended Pipeline seperti main.py, dengan dukungan untuk:
    - Multiple observables (training)
    - Automatic dataset generation
    - ML model training dan testing
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        n_circuits: int = 2000,
        output_dir: str = "output"
    ):
        self.n_qubits = n_qubits
        self.n_circuits = n_circuits
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Components
        config = AnsatzConfig(n_qubits=n_qubits)
        self.ansatz_gen = GenerateAnsatz(config)
        self.energy_calc = CalculateEnergy(n_qubits)
        self.ml_trainer = TrainML()
        
        logger.info(f"Extended Pipeline initialized: {n_circuits} circuits, {n_qubits} qubits")
    
    def generate_training_dataset(
        self,
        observable_list: List[str],
        model_id: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """
        Generate training dataset dengan multiple observables.
        
        Parameters
        ----------
        observable_list : List[str]
            List of observable Pauli strings
        model_id : int
            Model identifier
        
        Returns
        -------
        Tuple
            (noisy_energies, ideal_energies, observables, parameters)
        """
        logger.info(f"Generating dataset: {self.n_circuits} circuits, {len(observable_list)} observables")
        
        base_ansatz = self.ansatz_gen.create_ansatz()
        n_params = base_ansatz.num_parameters
        
        noisy_energies = []
        ideal_energies = []
        zne_energies = []
        observables = []
        parameters = []
        
        n_obs = len(observable_list)
        
        for idx in range(self.n_circuits):
            # Round-robin selection
            pauli = observable_list[idx % n_obs]
            obs_op = SparsePauliOp(pauli)
            
            # Random parameters
            theta = np.random.uniform(-np.pi, np.pi, size=n_params)
            
            # Calculate energies
            ideal_e = self.energy_calc.calculate_ideal_energy(base_ansatz, obs_op, theta)
            noisy_e = self.energy_calc.calculate_noisy_energy(base_ansatz, obs_op, theta)
            zne_e = self.energy_calc.calculate_zne_energy(base_ansatz, obs_op, theta)
            
            ideal_energies.append(float(ideal_e))
            noisy_energies.append(float(noisy_e))
            zne_energies.append(float(zne_e))
            observables.append(pauli)
            parameters.append(theta)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"{idx+1}/{self.n_circuits} circuits processed")
        
        # Convert to numpy arrays
        noisy_energies = np.array(noisy_energies, dtype=float)
        ideal_energies = np.array(ideal_energies, dtype=float)
        zne_energies = np.array(zne_energies, dtype=float)
        parameters = np.array(parameters, dtype=float)
        
        # Save to CSV
        self._save_dataset_csv(
            noisy_energies, ideal_energies, zne_energies,
            observables, parameters, model_id
        )
        
        logger.info(f"✓ Dataset generation completed")
        return noisy_energies, ideal_energies, observables, parameters
    
    def _save_dataset_csv(
        self,
        noisy_energies: np.ndarray,
        ideal_energies: np.ndarray,
        zne_energies: np.ndarray,
        observables: List[str],
        parameters: np.ndarray,
        model_id: int
    ):
        """Save dataset to CSV."""
        rows = []
        
        for i in range(len(noisy_energies)):
            row = {
                "noisy_energy": noisy_energies[i],
                "ideal_energy": ideal_energies[i],
                "zne_energy": zne_energies[i],
                "observable": observables[i]
            }
            
            for j, p in enumerate(parameters[i]):
                row[f"param_{j}"] = p
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, f"train_data/train_data_{model_id}.csv")
        df.to_csv(csv_path, index=False)
        
        logger.info(f"✓ Dataset saved to: {csv_path}")
    
    def train_model(
        self,
        noisy_energies: np.ndarray,
        ideal_energies: np.ndarray,
        observables: List[str],
        parameters: np.ndarray,
        model_id: int = 1,
        n_estimators: int = 100
    ) -> Dict:
        """Train ML model."""
        logger.info("Training ML model...")
        
        # Build dataset
        X, y = self.ml_trainer.build_dataset(
            noisy_energies, ideal_energies, observables, parameters
        )
        
        # Train
        metrics = self.ml_trainer.train_model(X, y, n_estimators=n_estimators)
        
        # Save model
        model_path = os.path.join(self.output_dir, f"ml_model/qem_model_{model_id}.pkl")
        self.ml_trainer.save_model(model_path)
        
        logger.info(f"✓ Model trained and saved to: {model_path}")
        return metrics
    
    def evaluate_model(
        self,
        test_data_path: str,
        model_id: int = 1
    ) -> pd.DataFrame:
        """Evaluate model on test data."""
        logger.info(f"Evaluating model on: {test_data_path}")
        
        test_data = pd.read_csv(test_data_path)
        
        noisy_energies = np.array(test_data['noisy_energy'], dtype=float)
        ideal_energies = np.array(test_data['ideal_energy'], dtype=float)
        observables = test_data['observable'].astype(str).tolist()
        
        # Extract parameters
        n_params = self.ansatz_gen.ansatz.num_parameters
        param_columns = [f'param_{i}' for i in range(n_params)]
        parameters = test_data[param_columns].to_numpy()
        
        # Build features
        X_test, _ = self.ml_trainer.build_dataset(
            noisy_energies, ideal_energies, observables, parameters
        )
        
        # Predict
        predictions = self.ml_trainer.predict(X_test)
        test_data['RF_energy'] = predictions
        
        # Save predictions
        pred_path = os.path.join(self.output_dir, f"predict_data/predicted_data_{model_id}.csv")
        test_data.to_csv(pred_path, index=False)
        
        logger.info(f"✓ Predictions saved to: {pred_path}")
        return test_data


# ============================================================================
# 6. FULL VQE PIPELINE
# ============================================================================

class VQEPipeline:
    """Orchestrate full ML-QEM VQE pipeline."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        output_dir: str = "qem_output"
    ):
        self.n_qubits = n_qubits
        self.output_dir = output_dir
        
        config = AnsatzConfig(n_qubits=n_qubits)
        self.ansatz_gen = GenerateAnsatz(config)
        self.energy_calc = CalculateEnergy(n_qubits)
        self.ml_trainer = TrainML()
        
        logger.info(f"VQE Pipeline initialized: {n_qubits} qubits")
    
    def run_full_vqe_pipeline(
        self,
        hamiltonian: SparsePauliOp,
        ml_model_path: Optional[str] = None,
        maxiter: int = 125
    ) -> Dict:
        """
        Run full VQE pipeline dengan ideal, noisy, ZNE, dan ML-mitigated.
        """
        logger.info("=" * 70)
        logger.info("QEM FULL VQE PIPELINE")
        logger.info("=" * 70)
        
        # Create ansatz
        ansatz = self.ansatz_gen.create_ansatz()
        logger.info(f"Ansatz info: {self.ansatz_gen.get_info()}")
        
        # Run VQE dengan semua backends
        vqe_runner = VQERunner(
            n_qubits=self.n_qubits,
            maxiter=maxiter,
            ml_model_path=ml_model_path
        )
        
        results = vqe_runner.run_all(ansatz, hamiltonian)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """Save results ke CSV pakai pandas."""
        rows = []

        for key, res in results.items():
            if res.get("status") == "skipped":
                rows.append({
                    "key": key,
                    "status": "skipped",
                    "energy": None,
                    "method": None,
                })
            else:
                rows.append({
                    "key": key,
                    "status": res.get("status", "done"),
                    "energy": res.get("energy"),
                    "method": res.get("method", "optimization"),
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir, index=False)

        logger.info(f"✓ Results saved to {self.output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML-QEM Pipeline with ZNE, H2 Hamiltonian, and Extended Features"
    )
    
    parser.add_argument("--n-qubits", type=int, default=4,
                        help="Number of qubits")
    parser.add_argument("--maxiter", type=int, default=125,
                        help="Max VQE iterations")
    parser.add_argument("--seed", type=int, default=170,
                        help="Random seed")
    parser.add_argument("--bond-length", type=float, default=0.735,
                        help="H2 bond length (Angstrom)")
    parser.add_argument("--model_path", type=str, default="output/ml_model/qem_model_1.pkl",
                        help="H2 bond length (Angstrom)")
    parser.add_argument("--output-dir", type=str, default="output/vqe_data/vqe_data.csv",
                        help="Output directory")

    
    args = parser.parse_args()
    
    # Generate H2 Hamiltonian
    h2_gen = H2HamiltonianGenerator(n_qubits=args.n_qubits)
    hamiltonian = h2_gen.generate_h2_hamiltonian(args.bond_length)
    
    # Create and run pipeline
    pipeline = VQEPipeline(
        n_qubits=args.n_qubits,
        output_dir=args.output_dir,
    )
    
    results = pipeline.run_full_vqe_pipeline(
        hamiltonian=hamiltonian,
        maxiter=args.maxiter,
        ml_model_path=args.model_path

    )
    
    # Print results
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
