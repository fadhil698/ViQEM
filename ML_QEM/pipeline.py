"""
================================================================================
Pipeline.py - Unified ML-QEM Pipeline dengan Extended dan VQE Functionality
================================================================================

Fitur:
- ExtendedPipeline: Training dataset, ML model training/evaluation
- VQEPipeline: Single & Multiple bond lengths VQE execution
- Support untuk Covalent workflow integration

Struktur:
1. ExtendedPipeline: Dataset generation, ML training
2. VQEPipeline: VQE execution (single & multi bond lengths)
3. Main utilities & configuration

Ready untuk integrasi dengan Covalent!
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel  # Pastikan module ini sudah terinstall

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# IMPORT DEPENDENCIES FROM qem_main_updated_v2
# ============================================================================

from ML_QEM.qem_main_updated_v2 import (
    H2HamiltonianGenerator,
    AnsatzConfig,
    GenerateAnsatz,
    CalculateEnergy,
    TrainML,
    VQERunner,
)


# ============================================================================
# 1. EXTENDED PIPELINE - Training & Model Management
# ============================================================================

class ExtendedPipeline:
    """
    Extended Pipeline untuk:
    - Generate training dataset dengan multiple observables
    - Train ML model (RandomForest)
    - Evaluate model pada test data
    
    Workflow:
    1. generate_training_dataset() -> CSV dengan (noisy, ideal, observable, params)
    2. train_model() -> Train RF model & save
    3. evaluate_model() -> Test pada new data
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_circuits: int = 2000,
        output_dir: str = "output",
        noise_model: Optional[NoiseModel] = None  # <--- TAMBAHAN DI SINI
        
    ):
        """
        Initialize ExtendedPipeline.
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits (default 4)
        n_circuits : int
            Number of circuits untuk training dataset (default 2000)
        output_dir : str
            Output directory untuk hasil (default "output")
        """
        self.n_qubits = n_qubits
        self.n_circuits = n_circuits
        self.output_dir = output_dir
        self.noise_model = noise_model
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "train_data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ml_model"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "predict_data"), exist_ok=True)
        
        # Components
        config = AnsatzConfig(n_qubits=n_qubits)
        self.ansatz_gen = GenerateAnsatz(config)
        self.energy_calc = CalculateEnergy(n_qubits, noise_model=noise_model)
        self.ml_trainer = TrainML()
        
        logger.info(f"ExtendedPipeline initialized: {n_circuits} circuits, {n_qubits} qubits")

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
            List of observable Pauli strings (misal ["IIII", "IIIZ", "IIZZ"])
        model_id : int
            Model identifier untuk file naming
            
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
            theta = np.random.uniform(-5.0, 5.0, size=n_params)
            
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
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
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
        """
        Train ML model (RandomForest).
        
        Parameters
        ----------
        noisy_energies : np.ndarray
            Noisy energy measurements
        ideal_energies : np.ndarray
            Ideal/ground truth energies
        observables : List[str]
            Observable Pauli strings
        parameters : np.ndarray
            Circuit parameters
        model_id : int
            Model identifier
        n_estimators : int
            Number of trees untuk RandomForest
            
        Returns
        -------
        Dict
            Training metrics (MSE, R², etc.)
        """
        logger.info("Training ML model...")
        
        # Build dataset
        X, y = self.ml_trainer.build_dataset(
            noisy_energies, ideal_energies, observables, parameters
        )
        
        # Train
        metrics = self.ml_trainer.train_model(X, y, n_estimators=n_estimators)
        
        # Save model
        model_path = os.path.join(self.output_dir, f"ml_model/qem_model_{model_id}.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.ml_trainer.save_model(model_path)
        
        logger.info(f"✓ Model trained and saved to: {model_path}")
        return metrics

    def evaluate_model(
        self,
        test_data_path: str,
        model_id: int = 1
    ) -> pd.DataFrame:
        """
        Evaluate model on test data.
        
        Parameters
        ----------
        test_data_path : str
            Path ke test data CSV
        model_id : int
            Model identifier
            
        Returns
        -------
        pd.DataFrame
            Predictions dengan columns: original data + RF_energy
        """
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
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        test_data.to_csv(pred_path, index=False)
        
        logger.info(f"✓ Predictions saved to: {pred_path}")
        return test_data


# ============================================================================
# 2. VQE PIPELINE - VQE Execution (Single & Multiple Bond Lengths)
# ============================================================================

class VQEPipeline:
    """
    Orchestrate full ML-QEM VQE pipeline dengan support untuk:
    - Single bond length VQE
    - Multiple bond lengths VQE (ideal untuk Covalent integration)
    - Ideal, Noisy, ZNE, dan ML-mitigated backends
    """

    def __init__(
        self,
        n_qubits: int = 4,
        output_dir: str = "qem_output",
        noise_model: Optional[NoiseModel] = None  # <--- TAMBAHAN DI SINI
    ):
        """
        Initialize VQEPipeline.
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits
        output_dir : str
            Output directory untuk hasil
        """
        self.n_qubits = n_qubits
        self.output_dir = output_dir
        self.noise_model = noise_model
        os.makedirs(output_dir, exist_ok=True)
        
        # Components
        config = AnsatzConfig(n_qubits=n_qubits)
        self.ansatz_gen = GenerateAnsatz(config)
        self.energy_calc = CalculateEnergy(n_qubits, noise_model=noise_model)
        self.ml_trainer = TrainML()
        self.h2_gen = H2HamiltonianGenerator(n_qubits=n_qubits)
        
        logger.info(f"VQEPipeline initialized: {n_qubits} qubits")

    def run_full_vqe_pipeline(
        self,
        hamiltonian: SparsePauliOp,
        ml_model_path: Optional[str] = None,
        maxiter: int = 125
    ) -> Dict:
        """
        Run full VQE pipeline untuk SATU Hamiltonian.
        
        Parameters
        ----------
        hamiltonian : SparsePauliOp
            Hamiltonian untuk satu bond length
        ml_model_path : Optional[str]
            Path ke ML model untuk mitigation
        maxiter : int
            Max iterasi untuk VQE optimizer
            
        Returns
        -------
        Dict
            Hasil VQE: {ideal, noisy, zne, mitigated}
        """
        logger.info("=" * 70)
        logger.info("QEM FULL VQE PIPELINE (SINGLE BOND LENGTH)")
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

    def run_full_vqe_pipeline_multi_bond_lengths(
        self,
        bond_lengths: List[float],
        ml_model_path: Optional[str] = None,
        maxiter: int = 125,
        save_results: bool = True
    ) -> Dict[float, Dict]:
        """
        Run full VQE pipeline untuk MULTIPLE bond lengths.
        
        🎯 FITUR UTAMA:
        - Input: List[float] bond_lengths (misal [0.5, 0.6, 0.735, 1.0, 1.5])
        - Auto generate Hamiltonian untuk setiap bond length
        - Jalankan full VQE (ideal, noisy, ZNE, mitigated) untuk masing-masing
        - Return Dict[bond_length] untuk mudah diakses hasil per bond length
        
        Parameters
        ----------
        bond_lengths : List[float]
            List of bond lengths dalam Angstrom
        ml_model_path : Optional[str]
            Path ke ML model untuk error mitigation
        maxiter : int
            Max iterasi optimizer (default 125)
        save_results : bool
            Jika True, simpan hasil ke CSV
            
        Returns
        -------
        Dict[float, Dict]
            {
                0.5: {
                    'ideal': {'energy': ..., 'method': ...},
                    'noisy': {'energy': ..., 'history': ...},
                    'zne': {'energy': ..., 'history': ...},
                    'mitigated': {'energy': ..., 'history': ...}
                },
                0.6: {...},
                ...
            }
            
        Example
        -------
        >>> pipeline = VQEPipeline(n_qubits=4)
        >>> bond_lengths = [0.5, 0.735, 1.0, 1.5]
        >>> results = pipeline.run_full_vqe_pipeline_multi_bond_lengths(
        ...     bond_lengths=bond_lengths,
        ...     ml_model_path="output/ml_model/qem_model_1.pkl",
        ...     maxiter=125
        ... )
        >>> 
        >>> # Akses hasil untuk bond_length 0.735:
        >>> energy_noisy = results[0.735]['noisy']['energy']
        >>> energy_zne = results[0.735]['zne']['energy']
        >>> energy_ideal = results[0.735]['ideal']['energy']
        """
        
        logger.info("=" * 70)
        logger.info("QEM FULL VQE PIPELINE (MULTIPLE BOND LENGTHS)")
        logger.info("=" * 70)
        logger.info(f"Bond lengths: {bond_lengths}")
        
        # Step 1: Generate Hamiltonian untuk semua bond lengths
        logger.info(f"\n[STEP 1] Generating Hamiltonians untuk {len(bond_lengths)} bond lengths...")
        bond_lengths_arr = np.array(bond_lengths, dtype=float)
        hamiltonians = self.h2_gen.generate_hamiltonian_range(bond_lengths_arr)
        logger.info(f"✓ {len(hamiltonians)} Hamiltonians generated")
        
        # Step 2: Create ansatz (shared untuk semua bond lengths)
        logger.info(f"\n[STEP 2] Creating ansatz...")
        ansatz = self.ansatz_gen.create_ansatz()
        ansatz_info = self.ansatz_gen.get_info()
        logger.info(f"✓ Ansatz created: {ansatz_info}")
        
        # Step 3: Initialize VQE runner
        logger.info(f"\n[STEP 3] Initializing VQE runner...")
        vqe_runner = VQERunner(
            n_qubits=self.n_qubits,
            maxiter=maxiter,
            ml_model_path=ml_model_path
        )
        logger.info(f"✓ VQE runner ready")
        
        # Step 4: Run full pipeline untuk SETIAP bond length
        results_all = {}
        total_bond_lengths = len(hamiltonians)
        
        logger.info(f"\n[STEP 4] Running VQE pipeline untuk setiap bond length...")
        logger.info(f"Total bond lengths: {total_bond_lengths}\n")
        
        for idx, (bond_length, hamiltonian) in enumerate(hamiltonians.items(), 1):
            logger.info(f"{'='*70}")
            logger.info(f"[{idx}/{total_bond_lengths}] Bond Length: {bond_length:.4f} Å")
            logger.info(f"{'='*70}")
            
            try:
                # Run VQE dengan semua backends untuk bond length ini
                results_bl = vqe_runner.run_all(ansatz, hamiltonian)
                results_all[bond_length] = results_bl
                
                # Summary untuk bond length ini
                logger.info(f"\n✓ Bond length {bond_length:.4f} completed:")
                logger.info(f"  - Ideal energy:    {results_bl['ideal'].get('energy', 'N/A'):.6f}")
                logger.info(f"  - Noisy energy:    {results_bl['noisy'].get('energy', 'N/A'):.6f}")
                logger.info(f"  - ZNE energy:      {results_bl['zne'].get('energy', 'N/A'):.6f}")
                if results_bl['mitigated'].get('status') != 'skipped':
                    logger.info(f"  - Mitigated energy: {results_bl['mitigated'].get('energy', 'N/A'):.6f}")
                
            except Exception as e:
                logger.error(f"✗ Error processing bond_length {bond_length:.4f}: {e}")
                results_all[bond_length] = {"status": "error", "error": str(e)}
        
        # Step 5: Save results jika diminta
        if save_results:
            logger.info(f"\n[STEP 5] Saving results to CSV...")
            self._save_multi_bond_length_results(results_all)
            logger.info(f"✓ Results saved")
        
        # Step 6: Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY - VQE Results untuk Multiple Bond Lengths")
        logger.info(f"{'='*70}")
        
        summary_rows = []
        for bl in sorted(results_all.keys()):
            res = results_all[bl]
            if res.get('status') == 'error':
                summary_rows.append({
                    'bond_length': f"{bl:.4f}",
                    'ideal': 'ERROR',
                    'noisy': 'ERROR',
                    'zne': 'ERROR',
                    'mitigated': 'ERROR'
                })
            else:
                summary_rows.append({
                    'bond_length': f"{bl:.4f}",
                    'ideal': f"{res['ideal'].get('energy', np.nan):.6f}",
                    'noisy': f"{res['noisy'].get('energy', np.nan):.6f}",
                    'zne': f"{res['zne'].get('energy', np.nan):.6f}",
                    'mitigated': f"{res['mitigated'].get('energy', np.nan):.6f}" if res['mitigated'].get('status') != 'skipped' else "SKIPPED"
                })
        
        summary_df = pd.DataFrame(summary_rows)
        logger.info("\n" + summary_df.to_string(index=False))
        
        return results_all
    
    def _save_results(self, results: Dict):
        """Save results untuk single bond length ke CSV."""
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
        csv_path = os.path.join(self.output_dir, "vqe_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Results saved to {csv_path}")

    def _save_multi_bond_length_results(self, results_all: Dict[float, Dict]):
        """
        Save results untuk multiple bond lengths ke CSV.
        
        Output columns:
        bond_length, ideal_energy, noisy_energy, zne_energy, mitigated_energy, status
        """
        rows = []
        for bond_length in sorted(results_all.keys()):
            res = results_all[bond_length]
            
            if res.get('status') == 'error':
                rows.append({
                    'bond_length': float(bond_length),
                    'ideal_energy': np.nan,
                    'noisy_energy': np.nan,
                    'zne_energy': np.nan,
                    'mitigated_energy': np.nan,
                    'status': 'error',
                    'error_msg': res.get('error', 'Unknown error')
                })
            else:
                ideal_e = res.get('ideal', {}).get('energy', np.nan)
                noisy_e = res.get('noisy', {}).get('energy', np.nan)
                zne_e = res.get('zne', {}).get('energy', np.nan)
                
                mitigated_status = res.get('mitigated', {}).get('status')
                if mitigated_status == 'skipped':
                    mitigated_e = np.nan
                else:
                    mitigated_e = res.get('mitigated', {}).get('energy', np.nan)
                
                rows.append({
                    'bond_length': float(bond_length),
                    'ideal_energy': ideal_e,
                    'noisy_energy': noisy_e,
                    'zne_energy': zne_e,
                    'mitigated_energy': mitigated_e,
                    'status': 'done' if mitigated_status != 'skipped' else 'mitigated_skipped',
                    'error_msg': None
                })
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, "vqe_results_multi_bond_length.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"✓ Multi-bond-length results saved to: {csv_path}")
        logger.info("\nResults Table:")
        logger.info(df.to_string(index=False))


# ============================================================================
# 3. UTILITY FUNCTIONS & MAIN
# ============================================================================

def create_h2_hamiltonian(bond_length: float, n_qubits: int = 4) -> SparsePauliOp:
    """
    Quick utility untuk generate H2 Hamiltonian.
    
    Parameters
    ----------
    bond_length : float
        Bond length dalam Angstrom
    n_qubits : int
        Number of qubits
        
    Returns
    -------
    SparsePauliOp
        H2 Hamiltonian
    """
    h2_gen = H2HamiltonianGenerator(n_qubits=n_qubits)
    return h2_gen.generate_h2_hamiltonian(bond_length)


if __name__ == "__main__":
    """
    Example usage untuk testing
    """
    
    # ========== EXAMPLE 1: Single Bond Length VQE ==========
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Bond Length VQE")
    print("="*70)
    
    # Create H2 Hamiltonian for bond_length = 0.735 Å
    h2_gen = H2HamiltonianGenerator(n_qubits=4)
    hamiltonian = h2_gen.generate_h2_hamiltonian(0.735)
    
    # Run VQE
    pipeline_single = VQEPipeline(n_qubits=4, output_dir="output/vqe_single")
    results_single = pipeline_single.run_full_vqe_pipeline(
        hamiltonian=hamiltonian,
        maxiter=50  # Quick test
    )
    
    print("\nSingle Bond Length Results:")
    for key, res in results_single.items():
        if res.get("status") != "skipped":
            print(f"  {key}: {res.get('energy'):.6f}")
    
    # ========== EXAMPLE 2: Multiple Bond Lengths VQE ==========
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Bond Lengths VQE")
    print("="*70)
    
    bond_lengths = [0.5, 0.735, 1.0]
    
    pipeline_multi = VQEPipeline(n_qubits=4, output_dir="output/vqe_multi")
    results_multi = pipeline_multi.run_full_vqe_pipeline_multi_bond_lengths(
        bond_lengths=bond_lengths,
        maxiter=50  # Quick test
    )
    
    print("\nMultiple Bond Lengths Results:")
    for bl in sorted(results_multi.keys()):
        if results_multi[bl].get('status') != 'error':
            print(f"  Bond length {bl:.4f}:")
            print(f"    Ideal:  {results_multi[bl]['ideal']['energy']:.6f}")
            print(f"    Noisy:  {results_multi[bl]['noisy']['energy']:.6f}")
            print(f"    ZNE:    {results_multi[bl]['zne']['energy']:.6f}")
    
    # ========== EXAMPLE 3: ExtendedPipeline - Training Dataset ==========
    print("\n" + "="*70)
    print("EXAMPLE 3: ExtendedPipeline - Training Dataset")
    print("="*70)
    
    ext_pipeline = ExtendedPipeline(n_qubits=4, n_circuits=100, output_dir="output/extended")
    
    # Generate training dataset
    observable_list = ["IIII", "IIIZ", "IIZZ", "ZZZZ"]
    noisy, ideal, obs, params = ext_pipeline.generate_training_dataset(
        observable_list=observable_list,
        model_id=1
    )
    
    print(f"\nDataset Generated:")
    print(f"  Noisy energies shape: {noisy.shape}")
    print(f"  Ideal energies shape: {ideal.shape}")
    print(f"  Observables: {len(obs)} samples")
    
    # Train ML model
    metrics = ext_pipeline.train_model(
        noisy_energies=noisy,
        ideal_energies=ideal,
        observables=obs,
        parameters=params,
        model_id=1,
        n_estimators=10  # Quick test
    )
    
    print(f"\nML Model Metrics:")
    for key, val in metrics.items():
        print(f"  {key}: {val:.6f}")
