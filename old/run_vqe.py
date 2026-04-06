"""
run_vqe.py - VQE dengan hybrid execution (ideal, noisy, ML-mitigated)

Jalankan VQE optimization pada Hamiltonian dengan tiga backend:
1. Ideal (statevector, no noise)
2. Noisy (with noise model)
3. ML-mitigated (noisy + RandomForest correction)

Usage:
    python run_vqe.py --hamiltonian H2 --maxiter 100
    python run_vqe.py --hamiltonian H2 --maxiter 100 --load-model qem_model.pkl
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime
import pickle
import joblib

from qiskit.circuit.library import NLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import Layout
from qiskit_ibm_runtime.fake_provider import FakeAthensV2
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_algorithms.optimizers import SPSA, OptimizerResult
from qiskit_algorithms.utils import algorithm_globals
from tqdm import tqdm
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"vqe_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VQERunner:
    """
    Menjalankan VQE dengan tiga skenario: ideal, noisy, dan ML-mitigated.
    """
    
    def __init__(
        self,
        n_qubits: int = 4,
        maxiter: int = 125,
        backend_fake_class=FakeAthensV2,
        seed: int = 170,
        ml_model_path: Optional[str] = None
    ):
        """
        Parameters
        ----------
        n_qubits : int
            Jumlah qubit
        maxiter : int
            Jumlah iterasi SPSA
        backend_fake_class
            Fake backend untuk setup noise model (default: FakeAthensV2)
        seed : int
            Random seed untuk reproducibility
        ml_model_path : str, optional
            Path ke model RandomForest yang sudah dilatih
        """
        self.n_qubits = n_qubits
        self.maxiter = maxiter
        self.seed = seed
        
        algorithm_globals.random_seed = seed
        
        # Setup estimators
        self.noiseless_estimator = Estimator(options={"default_precision": 1e-2})
        
        # Setup noisy estimator
        backend_fake = backend_fake_class()
        noise_model = NoiseModel.from_backend(backend_fake)
        self.noisy_estimator = Estimator(
            options={"default_precision": 1e-2},
            backend_options={"noise_model": noise_model}
        )
        
        self.backend_fake = backend_fake
        
        # Load ML model jika ada
        self.ml_model = None
        if ml_model_path and os.path.exists(ml_model_path):
            logger.info(f"Loading ML model from {ml_model_path}")
            self.ml_model = joblib.load(ml_model_path)
            logger.info(f"Model loaded. Features: {self.ml_model.n_features_in_}")
        
        # Untuk tracking hasil
        self.history = {
            "ideal": {"counts": [], "values": [], "thetas": []},
            "noisy": {"counts": [], "values": [], "thetas": []},
            "mitigated": {"counts": [], "values": [], "thetas": []}
        }
    
    def create_ansatz(self) -> object:
        """Buat hardware-efficient ansatz."""
        ansatz = NLocal(
            self.n_qubits,
            rotation_blocks="ry",
            entanglement_blocks="cx",
            entanglement="linear",
            reps=2,
            insert_barriers=True
        )
        return ansatz
    
    def run_vqe_ideal(
        self,
        hamiltonian: SparsePauliOp,
        ansatz,
    ) -> Dict:
        """
        Jalankan VQE dengan ideal backend (no noise).
        """
        logger.info("=" * 70)
        logger.info("STAGE 1: VQE IDEAL (Noiseless)")
        logger.info("=" * 70)
        
        history_ideal = {"counts": [], "values": [], "thetas": []}
        
        def cost_fn(theta):
            nonlocal history_ideal
            evalcount = len(history_ideal["counts"]) + 1
            
            theta_arr = np.asarray(theta, dtype=float)
            pub = (ansatz, hamiltonian, theta_arr)
            job = self.noiseless_estimator.run([pub])
            result = job.result()
            
            energy = float(result[0].data.evs[0])
            
            history_ideal["counts"].append(evalcount)
            history_ideal["values"].append(energy)
            history_ideal["thetas"].append(theta)
            
            if evalcount % 10 == 0:
                logger.info(f"  Ideal Iteration {evalcount}: E = {energy:.6f}")
            
            return energy
        
        # Initial point
        num_params = ansatz.num_parameters
        x0 = 2 * np.pi * np.random.random(num_params)
        
        # Optimize dengan SPSA
        optimizer = SPSA(maxiter=self.maxiter)
        
        logger.info(f"Starting optimization (maxiter={self.maxiter})...")
        result = optimizer.minimize(cost_fn, x0=x0)
        
        logger.info(f"✓ Ideal VQE completed.")
        logger.info(f"  Final energy: {result.fun:.6f}")
        logger.info(f"  Evaluations: {len(history_ideal['counts'])}")
        
        self.history["ideal"] = history_ideal
        
        return {
            "energy_min": result.fun,
            "theta_opt": result.x,
            "history": history_ideal,
            "evaluations": len(history_ideal["counts"])
        }
    
    def run_vqe_noisy(
        self,
        hamiltonian: SparsePauliOp,
        ansatz,
    ) -> Dict:
        """
        Jalankan VQE dengan noisy backend (tanpa mitigation).
        """
        logger.info("=" * 70)
        logger.info("STAGE 2: VQE NOISY (With noise model, no mitigation)")
        logger.info("=" * 70)
        
        history_noisy = {"counts": [], "values": [], "thetas": []}
        
        def cost_fn(theta):
            nonlocal history_noisy
            evalcount = len(history_noisy["counts"]) + 1
            
            theta_arr = np.asarray(theta, dtype=float)
            pub = (ansatz, hamiltonian, theta_arr)
            job = self.noisy_estimator.run([pub])
            result = job.result()
            
            energy = float(result[0].data.evs[0])
            
            history_noisy["counts"].append(evalcount)
            history_noisy["values"].append(energy)
            history_noisy["thetas"].append(theta)
            
            if evalcount % 10 == 0:
                logger.info(f"  Noisy Iteration {evalcount}: E = {energy:.6f}")
            
            return energy
        
        # Initial point
        num_params = ansatz.num_parameters
        algorithm_globals.random_seed = self.seed
        x0 = 2 * np.pi * np.random.random(num_params)
        
        # Optimize
        optimizer = SPSA(maxiter=self.maxiter)
        
        logger.info(f"Starting optimization (maxiter={self.maxiter})...")
        result = optimizer.minimize(cost_fn, x0=x0)
        
        logger.info(f"✓ Noisy VQE completed.")
        logger.info(f"  Final energy: {result.fun:.6f}")
        logger.info(f"  Evaluations: {len(history_noisy['counts'])}")
        
        self.history["noisy"] = history_noisy
        
        return {
            "energy_min": result.fun,
            "theta_opt": result.x,
            "history": history_noisy,
            "evaluations": len(history_noisy["counts"])
        }
    
    def run_vqe_mitigated(
        self,
        hamiltonian: SparsePauliOp,
        ansatz,
    ) -> Dict:
        """
        Jalankan VQE dengan ML mitigation.
        """
        if self.ml_model is None:
            logger.warning("ML model not loaded. Skipping mitigated VQE.")
            return {"status": "skipped", "reason": "ML model not available"}
        
        logger.info("=" * 70)
        logger.info("STAGE 3: VQE ML-MITIGATED (Noisy + RandomForest correction)")
        logger.info("=" * 70)
        
        history_mitigated = {"counts": [], "values": [], "thetas": []}
        
        # Ambil feature names dari model
        feature_names = list(self.ml_model.feature_names_in_)
        param_features = [n for n in feature_names if n.startswith("param")]
        n_params_rf = len(param_features)
        
        logger.info(f"RF model features: {len(feature_names)}")
        logger.info(f"RF expects {n_params_rf} parameters")
        
        if ansatz.num_parameters < n_params_rf:
            logger.error(f"Ansatz hanya punya {ansatz.num_parameters} parameters "
                        f"tapi RF model mengharapkan {n_params_rf}")
            return {"status": "error", "reason": "Parameter mismatch"}
        
        # Hitung gate counts
        layout = Layout([(i, i) for i in range(self.n_qubits)])
        pm = generate_preset_pass_manager(
            backend=self.backend_fake,
            optimization_level=0,
            seed_transpiler=self.seed,
            initial_layout=layout
        )
        ansatz_transpiled = pm.run(ansatz)
        op_counts = ansatz_transpiled.count_ops()
        
        num_x = int(op_counts.get("x", 0))
        num_sx = int(op_counts.get("sx", 0))
        num_cx = int(op_counts.get("cx", 0))
        
        logger.info(f"Gate counts: X={num_x}, SX={num_sx}, CX={num_cx}")
        
        # Observable encoder
        def encode_observable(label: str) -> Dict:
            """One-hot encode Pauli string."""
            feat = {}
            for pos in range(self.n_qubits):
                for op in ["I", "X", "Y", "Z"]:
                    feat[f"obs{pos}{op}"] = 0
            
            for pos, ch in enumerate(label):
                feat[f"obs{pos}{ch}"] = 1
            
            return feat
        
        # Build feature row untuk satu term Pauli
        def build_feature_row(theta, noisy_exp: float, label: str) -> List:
            """Build feature row sesuai urutan feature_names model."""
            theta_arr = np.asarray(theta, dtype=float)
            
            feat = {}
            
            # 1. Noisy exp value
            feat["Noisy exp"] = float(noisy_exp)
            
            # 2. Parameters
            for i, pname in enumerate(param_features):
                feat[pname] = float(theta_arr[i])
            
            # 3. Gate counts (konstan)
            feat["numX"] = num_x
            feat["numSX"] = num_sx
            feat["numCX"] = num_cx
            
            # 4. Observable one-hot
            feat.update(encode_observable(label))
            
            # 5. Susun sesuai urutan feature_names
            row = [feat.get(name, 0.0) for name in feature_names]
            
            return row
        
        # Cost function dengan RF mitigation
        def cost_fn(theta):
            nonlocal history_mitigated
            evalcount = len(history_mitigated["counts"]) + 1
            
            theta_arr = np.asarray(theta, dtype=float)
            
            # Siapkan Hamiltonian
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
            
            # Siapkan PUB untuk setiap term
            for label, coeff in zip(labels, coeffs):
                if np.isclose(coeff, 0.0):
                    continue
                
                # Identity term
                if set(label) == {"I"}:
                    identity_shift = float(coeff)
                    continue
                
                obs = SparsePauliOp(label)
                pubs.append((ansatz, obs, theta_arr))
                active_labels.append(label)
                active_coeffs.append(float(coeff))
            
            if not pubs:
                # Hamiltonian hanya konstanta
                return identity_shift
            
            # Jalankan estimator
            job = self.noisy_estimator.run(pubs)
            results = job.result()
            
            # Build feature matrix dan dapatkan prediksi RF
            rows = []
            for res, label in zip(results, active_labels):
                vals = np.asarray(res.data.evs)
                noisy_exp = float(vals.ravel()[0])
                rows.append(build_feature_row(theta, noisy_exp, label))
            
            X_df = pd.DataFrame(rows, columns=feature_names)
            pred_terms = self.ml_model.predict(X_df)
            
            # Hitung energi
            energy = identity_shift + float(np.dot(active_coeffs, pred_terms))
            
            history_mitigated["counts"].append(evalcount)
            history_mitigated["values"].append(energy)
            history_mitigated["thetas"].append(theta)
            
            if evalcount % 10 == 0:
                logger.info(f"  Mitigated Iteration {evalcount}: E = {energy:.6f}")
            
            return energy
        
        # Initial point
        num_params = ansatz.num_parameters
        algorithm_globals.random_seed = self.seed
        x0 = 2 * np.pi * np.random.random(num_params)
        
        # Optimize
        optimizer = SPSA(maxiter=self.maxiter)
        
        logger.info(f"Starting ML-mitigated optimization (maxiter={self.maxiter})...")
        result = optimizer.minimize(cost_fn, x0=x0)
        
        logger.info(f"✓ ML-mitigated VQE completed.")
        logger.info(f"  Final energy: {result.fun:.6f}")
        logger.info(f"  Evaluations: {len(history_mitigated['counts'])}")
        
        self.history["mitigated"] = history_mitigated
        
        return {
            "energy_min": result.fun,
            "theta_opt": result.x,
            "history": history_mitigated,
            "evaluations": len(history_mitigated["counts"])
        }
    
    def run_all(self, hamiltonian: SparsePauliOp) -> Dict:
        """Jalankan ketiga VQE scenario."""
        ansatz = self.create_ansatz()
        
        results = {}
        
        # Ideal VQE
        results["ideal"] = self.run_vqe_ideal(hamiltonian, ansatz)
        
        # Noisy VQE
        results["noisy"] = self.run_vqe_noisy(hamiltonian, ansatz)
        
        # ML-mitigated VQE
        results["mitigated"] = self.run_vqe_mitigated(hamiltonian, ansatz)
        
        return results
    
    def plot_comparison(self, results: Dict, output_path: str = "vqe_comparison.png"):
        """Plot perbandingan ketiga skenario."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (key, res) in enumerate(results.items()):
            if res.get("status") == "skipped":
                axes[idx].text(0.5, 0.5, "Skipped", ha="center", va="center")
                axes[idx].set_title(key.upper())
                continue
            
            hist = res.get("history", {})
            if not hist.get("counts"):
                continue
            
            counts = hist["counts"]
            values = hist["values"]
            
            axes[idx].plot(counts, values, "o-", alpha=0.7)
            axes[idx].set_xlabel("Evaluation")
            axes[idx].set_ylabel("Energy")
            axes[idx].set_title(f"{key.upper()} (Final: {res['energy_min']:.6f})")
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        logger.info(f"Plot saved to {output_path}")
    
    def save_results(self, results: Dict, output_path: str = "vqe_results.json"):
        """Save results ke JSON (energy & evaluations)."""
        summary = {}
        for key, res in results.items():
            summary[key] = {
                "energy_min": res.get("energy_min"),
                "evaluations": res.get("evaluations"),
                "status": res.get("status", "completed")
            }
        
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run VQE with ideal, noisy, and ML-mitigated backends"
    )
    parser.add_argument("--n-qubits", type=int, default=4,
                       help="Number of qubits (default: 4)")
    parser.add_argument("--maxiter", type=int, default=125,
                       help="Max iterations SPSA (default: 125)")
    parser.add_argument("--hamiltonian", type=str, default="H2",
                       help="Hamiltonian type (H2, LiH, etc.)")
    parser.add_argument("--load-model", type=str, default=None,
                       help="Path to pre-trained RandomForest model")
    parser.add_argument("--output-dir", type=str, default="vqe_output",
                       help="Output directory (default: vqe_output)")
    parser.add_argument("--seed", type=int, default=170,
                       help="Random seed (default: 170)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info(f"VQE with Hybrid Execution")
    logger.info(f"N_qubits: {args.n_qubits}, MaxIter: {args.maxiter}")
    logger.info(f"Hamiltonian: {args.hamiltonian}")
    logger.info(f"ML Model: {args.load_model if args.load_model else 'None'}")
    logger.info("=" * 70)
    
    # Create simple Hamiltonian untuk testing
    # TODO: Replace dengan actual Hamiltonian dari qiskit-nature
    if args.hamiltonian == "H2":
        # Simplified H2 Hamiltonian (4 qubits)
        hamiltonian = SparsePauliOp(
            ["IIZZ", "IZZZ", "ZIZZ", "ZZIZ"],
            [0.1, 0.2, 0.15, 0.1]
        )
        logger.info("Using test H2-like Hamiltonian")
    else:
        logger.error(f"Unknown Hamiltonian: {args.hamiltonian}")
        return
    
    # Run VQE
    runner = VQERunner(
        n_qubits=args.n_qubits,
        maxiter=args.maxiter,
        seed=args.seed,
        ml_model_path=args.load_model
    )
    
    results = runner.run_all(hamiltonian)
    
    # Save results
    output_results = os.path.join(args.output_dir, "vqe_results.json")
    runner.save_results(results, output_results)
    
    # Plot
    output_plot = os.path.join(args.output_dir, "vqe_comparison.png")
    runner.plot_comparison(results, output_plot)
    
    # Print summary
    logger.info("=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    for key, res in results.items():
        if res.get("status") == "skipped":
            logger.info(f"{key.upper()}: SKIPPED ({res.get('reason')})")
        else:
            logger.info(f"{key.upper()}: E = {res.get('energy_min'):.6f}")
    
    logger.info("=" * 70)
    logger.info(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
