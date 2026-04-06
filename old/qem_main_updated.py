"""
qem_main_updated.py - ML-QEM Pipeline dengan NumPyMinimumEigensolver untuk Ideal VQE

Perubahan:
- run_vqe_ideal() sekarang menggunakan NumPyMinimumEigensolver (exact diagonalization)
  bukan SPSA optimization, jadi dapet exact ground truth energy
- run_vqe_noisy() tetap pakai SPSA + noisy estimator
- run_vqe_mitigated() tetap pakai SPSA + RF correction

Ini lebih efisien dan akurat untuk mendapatkan reference energy.
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
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import Layout
from qiskit_ibm_runtime.fake_provider import FakeAthensV2
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms import VQE, NumPyMinimumEigensolver
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Optional Covalent support
# try:
#     import covalent as ct
#     COVALENT_AVAILABLE = True
# except ImportError:
#     COVALENT_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. ANSATZ GENERATION
# ============================================================================

@dataclass
class AnsatzConfig:
    """Configuration untuk ansatz generation."""
    n_qubits: int = 4
    rotation_blocks: str = "ry"
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
            rotation_blocks=self.config.rotation_blocks,
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
# 2. ENERGY CALCULATION
# ============================================================================

@dataclass
class EnergyResult:
    """Hasil perhitungan energi."""
    ideal_energy: Optional[float] = None
    noisy_energy: Optional[float] = None
    zne_energy: Optional[float] = None
    theta_opt: Optional[np.ndarray] = None
    backend_name: str = "FakeAthensV2"
    noise_model: Optional[str] = None


class CalculateEnergy:
    """Hitung energi dengan multiple backends: ideal (exact), noisy, ZNE."""
    
    def __init__(self, n_qubits: int = 4, backend_class=FakeAthensV2):
        self.n_qubits = n_qubits
        self.backend_class = backend_class
        
        # Ideal estimator untuk noisy energy
        self.ideal_estimator = Estimator(options={"default_precision": 1e-2})
        
        # Noisy estimator
        backend_fake = backend_class()
        noise_model = NoiseModel.from_backend(backend_fake)
        
        # Zne estimator
        # masukkan ZNE estimator
        
        self.noisy_estimator = Estimator(
            options={
            "default_precision": 1e-2,
            "backend_options": {"noise_model": noise_model}
            }
        )
        self.backend_fake = backend_fake
        
        logger.info(f"Energy calculators initialized: ideal + noisy")
    
    def calculate_groundstate(
        self,
        hamiltonian: SparsePauliOp
    ) -> float:
        """
        Hitung ideal energy dengan exact diagonalization (NumPyMinimumEigensolver).
        Ini memberikan ground truth energy tanpa perlu optimization.
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
    
    def calculate_zne_energy(
        self,
        ansatz,
        hamiltonian: SparsePauliOp,
        theta: np.ndarray,
        noise_factors: List[float] = [1.0, 2.0, 3.0]
    ) -> float:
        """
        Hitung ZNE (Zero Noise Extrapolation) energy.
        Extrapolasi ke noise=0 dari multiple noise levels.
        """
        zne_energy = 0
        return zne_energy
    
    def calculate_all_energies(
        self,
        ansatz,
        hamiltonian: SparsePauliOp,
        theta: np.ndarray
    ) -> EnergyResult:
        """Hitung semua energi: ideal (exact), noisy, ZNE."""
        ideal = self.calculate_ideal_energy_exact(hamiltonian)
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
    
    def __init__(self, test_size: float = 1, random_state: int = 42):
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
        """
        Build feature matrix untuk training RF model.
        """
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
        logger.info("predicting model...")
        return self.model.predict(X)


# ============================================================================
# 4. VQE RUNNER (dengan ML mitigation)
# ============================================================================

class VQERunner:
    """Jalankan VQE dengan ideal (exact), noisy, dan ML-mitigated backends."""
    
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
            "mitigated": {"counts": [], "values": []}
        }
    
    def run_vqe_ideal(
        self,
        hamiltonian: SparsePauliOp
    ) -> Dict:
        """
        Jalankan ideal VQE dengan exact diagonalization.
        Menggunakan NumPyMinimumEigensolver, bukan SPSA optimization.
        """
        logger.info("=" * 70)
        logger.info("VQE IDEAL (Exact - NumPyMinimumEigensolver)")
        logger.info("=" * 70)
        
        try:
            # Hitung exact ground truth energy
            exact_energy = self.energy_calc.calculate_ideal_energy_exact(hamiltonian)
            
            self.history["ideal"]["value"] = exact_energy
            
            logger.info(f"✓ Ideal VQE completed (exact).")
            logger.info(f"  Ground truth energy: {exact_energy:.6f}")
            
            return {
                "energy": exact_energy,
                "method": "exact_diagonalization",
                "history": {"value": exact_energy}
            }
        
        except Exception as e:
            logger.error(f"Error in ideal VQE: {e}")
            raise
    
    def run_vqe_noisy(
        self,
        ansatz,
        hamiltonian: SparsePauliOp
    ) -> Dict:
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
                logger.info(f"  Noisy Iter {evalcount}: E = {energy:.6f}")
            
            return energy
        
        num_params = ansatz.num_parameters
        algorithm_globals.random_seed = self.seed
        x0 = 2 * np.pi * np.random.random(num_params)
        optimizer = SPSA(maxiter=self.maxiter)
        
        logger.info(f"Starting optimization ({self.maxiter} iterations)...")
        result = optimizer.minimize(cost_fn, x0=x0)
        
        logger.info(f"✓ Noisy VQE completed.")
        logger.info(f"  Final energy: {result.fun:.6f}")
        
        self.history["noisy"] = history
        
        return {
            "energy": result.fun,
            "theta_opt": result.x,
            "history": history
        }
    
    def run_vqe_mitigated(
        self,
        ansatz,
        hamiltonian: SparsePauliOp
    ) -> Dict:
        """Jalankan ML-mitigated VQE (noisy + RF correction)."""
        if self.ml_trainer.model is None:
            logger.warning("ML model not loaded. Skipping mitigated VQE.")
            return {"status": "skipped"}
        
        logger.info("=" * 70)
        logger.info("VQE ML-MITIGATED (Noisy + RF Correction)")
        logger.info("=" * 70)
        
        history = {"counts": [], "values": []}
        
        def cost_fn(theta):
            evalcount = len(history["counts"]) + 1
            
            # Compute energi dengan RF correction
            energy = self._compute_mitigated_energy(
                ansatz, hamiltonian, theta
            )
            
            history["counts"].append(evalcount)
            history["values"].append(energy)
            
            if evalcount % 10 == 0:
                logger.info(f"  Mitigated Iter {evalcount}: E = {energy:.6f}")
            
            return energy
        
        num_params = ansatz.num_parameters
        algorithm_globals.random_seed = self.seed
        x0 = 2 * np.pi * np.random.random(num_params)
        optimizer = SPSA(maxiter=self.maxiter)
        
        logger.info(f"Starting optimization ({self.maxiter} iterations)...")
        result = optimizer.minimize(cost_fn, x0=x0)
        
        logger.info(f"✓ Mitigated VQE completed.")
        logger.info(f"  Final energy: {result.fun:.6f}")
        
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
        theta: np.ndarray,
        num_x: int,
        num_sx: int,
        num_cx: int
    ) -> float:
        """Compute ML-corrected energy untuk satu Hamiltonian."""
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
            feat["noisy_exp"] = noisy_exp
            
            for i, p in enumerate(theta_arr):
                feat[f"param_{i}"] = float(p)
            
            feat["numX"] = num_x
            feat["numSX"] = num_sx
            feat["numCX"] = num_cx
            
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
        """Jalankan ketiga VQE scenario."""
        results = {}
        
        # Ideal: exact energy (no optimization)
        results["ideal"] = self.run_vqe_ideal(hamiltonian)
        
        # Noisy: SPSA optimization dengan noise
        results["noisy"] = self.run_vqe_noisy(ansatz, hamiltonian)
        
        # Mitigated: SPSA + RF correction
        results["mitigated"] = self.run_vqe_mitigated(ansatz, hamiltonian)
        
        return results


# ============================================================================
# 5. FULL PIPELINE ORCHESTRATION
# ============================================================================

class VQEPipeline:
    """Orchestrate full ML-QEM pipeline."""
    
    def __init__(
        self,
        n_qubits: int = 4,
        output_dir: str = "qem_output"
    ):
        self.n_qubits = n_qubits
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        ansatz_config = AnsatzConfig(n_qubits=n_qubits)
        self.ansatz_gen = GenerateAnsatz(ansatz_config)
        self.energy_calc = CalculateEnergy(n_qubits)
        self.ml_trainer = TrainML()
        
        logger.info(f"QEM Pipeline initialized: {n_qubits} qubits")
    
    def run_full_vqe_pipeline(
        self,
        hamiltonian: SparsePauliOp,
        ml_model_path: Optional[str] = None,
        maxiter: int = 125
    ) -> Dict:
        """
        Run full VQE pipeline dengan ideal (exact), noisy, dan ML-mitigated.
        
        Parameters
        ----------
        hamiltonian : SparsePauliOp
            Hamiltonian untuk VQE
        ml_model_path : str, optional
            Path ke pre-trained RF model
        maxiter : int
            Max SPSA iterations
        
        Returns
        -------
        Dict
            Results dari ketiga VQE scenario
        """
        logger.info("=" * 70)
        logger.info("QEM FULL VQE PIPELINE")
        logger.info("=" * 70)
        
        # 1. Create ansatz
        ansatz = self.ansatz_gen.create_ansatz()
        
        logger.info(f"Ansatz info: {self.ansatz_gen.get_info()}")
        
        # 2. Run VQE dengan semua backends
        vqe_runner = VQERunner(
            n_qubits=self.n_qubits,
            maxiter=maxiter,
            ml_model_path=ml_model_path
        )
        
        results = vqe_runner.run_all(ansatz, hamiltonian)
        
        # 3. Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict):
        """Save results ke JSON."""
        summary = {}
        for key, res in results.items():
            if res.get("status") == "skipped":
                summary[key] = {"status": "skipped"}
            else:
                summary[key] = {
                    "energy": res.get("energy"),
                    "method": res.get("method", "optimization")
                }
        
        output_file = os.path.join(self.output_dir, "vqe_results.json")
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Results saved to {output_file}")


# ============================================================================
# COVALENT WORKFLOW SUPPORT
# ============================================================================

# if COVALENT_AVAILABLE:
    
#     @ct.electron
#     def ideal_energy_task(hamiltonian_dict: dict) -> dict:
#         """Electron: Calculate ideal (exact) energy."""
#         labels = hamiltonian_dict["labels"]
#         coeffs = hamiltonian_dict["coeffs"]
#         hamiltonian = SparsePauliOp(labels, coeffs)
        
#         n_qubits = len(labels[0]) if labels else 4
#         calc = CalculateEnergy(n_qubits)
#         energy = calc.calculate_ideal_energy_exact(hamiltonian)
        
#         return {
#             "ideal_energy": energy,
#             "method": "exact_diagonalization"
#         }
    
#     @ct.electron
#     def vqe_noisy_task(
#         theta: np.ndarray,
#         hamiltonian_dict: dict,
#         n_qubits: int = 4,
#         maxiter: int = 100
#     ) -> dict:
#         """Electron: Run noisy VQE."""
#         labels = hamiltonian_dict["labels"]
#         coeffs = hamiltonian_dict["coeffs"]
#         hamiltonian = SparsePauliOp(labels, coeffs)
        
#         from qiskit.circuit.library import NLocal
#         ansatz = NLocal(
#             n_qubits,
#             rotation_blocks="ry",
#             entanglement_blocks="cx",
#             entanglement="linear",
#             reps=2
#         )
        
#         vqe = VQERunner(n_qubits=n_qubits, maxiter=maxiter)
#         result = vqe.run_vqe_noisy(ansatz, hamiltonian)
        
#         return result
    
#     @ct.lattice
#     def qem_vqe_workflow(
#         n_qubits: int = 4,
#         hamiltonian_dict: dict = None,
#         maxiter: int = 125
#     ) -> dict:
#         """Covalent workflow: Full QEM VQE pipeline."""
        
#         if hamiltonian_dict is None:
#             hamiltonian_dict = {
#                 "labels": ["IIZZ", "IZZZ", "ZIZZ", "ZZIZ"],
#                 "coeffs": [0.1, 0.2, 0.15, 0.1]
#             }
        
#         # Task 1: Ideal energy (exact)
#         ideal_result = ideal_energy_task(hamiltonian_dict)
        
#         # Task 2: Noisy VQE
#         theta_init = 2 * np.pi * np.random.random(n_qubits * 3 * 3)
#         noisy_result = vqe_noisy_task(
#             theta_init,
#             hamiltonian_dict,
#             n_qubits=n_qubits,
#             maxiter=maxiter
#         )
        
#         return {
#             "ideal": ideal_result,
#             "noisy": noisy_result
#         }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ML-QEM Pipeline with VQE"
    )
    parser.add_argument("--n-qubits", type=int, default=4,
                       help="Number of qubits")
    parser.add_argument("--maxiter", type=int, default=125,
                       help="Max VQE iterations (for noisy/mitigated)")
    parser.add_argument("--load-model", type=str, default=None,
                       help="Path to RF model")
    parser.add_argument("--output-dir", type=str, default="qem_output",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=170,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = VQEPipeline(
        n_qubits=args.n_qubits,
        output_dir=args.output_dir
    )
    
    # Simple test Hamiltonian (H2-like)
    hamiltonian = SparsePauliOp(
        ["IIZZ", "IZZZ", "ZIZZ", "ZZIZ"],
        [0.1, 0.2, 0.15, 0.1]
    )
    
    # Run VQE pipeline
    results = pipeline.run_full_vqe_pipeline(
        hamiltonian=hamiltonian,
        ml_model_path=args.load_model,
        maxiter=args.maxiter
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
