"""
================================================================================
UnifiedPipeline.py - End-to-End ML-QEM Workflow dengan Identifier
================================================================================

Workflow Terpadu:
1. Generate training dataset dengan multiple observables
2. Train ML model (RandomForest)
3. Evaluate model pada test data
4. Gunakan model untuk VQE execution (single/multi bond lengths)

Setiap run memiliki experiment_id yang organize output ke folder terpisah!
Filename tetap sama, cuma folder yang ter-label dengan ID.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel  # Pastikan module ini sudah terinstall

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import dari Pipeline.py
from ML_QEM.pipeline import VQEPipeline, ExtendedPipeline


# ============================================================================
# UNIFIED PIPELINE - ML Training → Evaluation → VQE dengan Identifier
# ============================================================================

class UnifiedPipeline:
    """
    Complete end-to-end pipeline:
    1. ML Training: Generate dataset + train model
    2. ML Evaluation: Test model accuracy
    3. VQE Execution: Use trained model untuk error mitigation

    Setiap run memiliki experiment_id yang mengorganisir output ke folder terpisah
    Filename tetap standar (tidak ada suffix ID).
    
    Workflow:
    ┌─────────────────────────────────────────────────────────┐
    │  Step 1: ML TRAINING                                    │
    │  ├─ Generate training data (multiple observables)       │
    │  ├─ Train RandomForest model                            │
    │  └─ Save qem_model.pkl (dalam folder experiment_id)    │
    ├─────────────────────────────────────────────────────────┤
    │  Step 2: ML EVALUATION                                  │
    │  ├─ Load test data                                      │
    │  ├─ Evaluate model performance                          │
    │  └─ Calculate metrics (MSE, R², etc.)                   │
    ├─────────────────────────────────────────────────────────┤
    │  Step 3: VQE EXECUTION                                  │
    │  ├─ Single bond length VQE                              │
    │  │ └─ Use trained model untuk mitigation                │
    │  └─ Multiple bond lengths VQE                           │
    │    └─ Use model untuk semua bond lengths                │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_circuits: int = 2000,
        output_dir: str = "unified_output_lupa",
        experiment_id: Optional[str] = None,
        noise_model: Optional[NoiseModel] = None  # <--- TAMBAHAN DI SINI
    ):
        """
        Initialize UnifiedPipeline dengan experiment identifier.
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits
        n_circuits : int
            Number of circuits untuk training dataset
        output_dir : str
            Main output directory
        experiment_id : Optional[str]
            Experiment identifier untuk organize output folder
            If None: auto-generated (experiment_YYYYMMDD_HHMMSS)
            Examples: "h2_vqe_exp", "exp_001", "lfim_test"
        """
        self.n_qubits = n_qubits
        self.n_circuits = n_circuits
        self.output_dir = output_dir
        self.noise_model = noise_model
        
        # Generate experiment_id jika tidak diberikan
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_id = f"experiment_{timestamp}"
        else:
            self.experiment_id = experiment_id
        
        # Create experiment-specific subdirectories (folder organize by ID)
        # But filenames stay the same!
        self.exp_dir = os.path.join(self.output_dir, self.experiment_id)
        self.ml_dir = os.path.join(self.exp_dir, "ml_training")
        self.eval_dir = os.path.join(self.exp_dir, "ml_evaluation")
        self.vqe_dir = os.path.join(self.exp_dir, "vqe_execution")
        
        for d in [self.ml_dir, self.eval_dir, self.vqe_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Initialize sub-pipelines
        self.extended_pipeline = ExtendedPipeline(
            n_qubits=n_qubits,
            n_circuits=n_circuits,
            output_dir=self.ml_dir,
            noise_model=noise_model
        )
        
        self.vqe_pipeline = VQEPipeline(
            n_qubits=n_qubits,
            output_dir=self.vqe_dir,
            noise_model=noise_model
        )
        
        # Store model path
        self.model_path = None
        self.model_metrics = None
        
        logger.info(f"UnifiedPipeline initialized: {n_qubits} qubits, {n_circuits} circuits")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Output directory: {self.exp_dir}")

    # ========================================================================
    # PHASE 1: ML TRAINING
    # ========================================================================
    
    def phase1_ml_training(
        self,
        observable_list: List[str],
        model_id: int = 1,
        n_estimators: int = 100
    ) -> Dict:
        """
        Phase 1: Generate training dataset dan train ML model.
        
        Parameters
        ----------
        observable_list : List[str]
            List of observable Pauli strings
            Example: ["IIII", "IIIZ", "IIZZ", "ZZZZ"]
        model_id : int
            Model identifier
        n_estimators : int
            RandomForest tree count
            
        Returns
        -------
        Dict
            Training results dengan paths dan metrics
        """
        logger.info("=" * 70)
        logger.info("PHASE 1: ML TRAINING")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info("=" * 70)
        
        logger.info(f"\n[1.1] Generating training dataset...")
        logger.info(f"  - Circuits: {self.n_circuits}")
        logger.info(f"  - Observables: {len(observable_list)}")
        logger.info(f"  - Observables: {observable_list}")
        
        # Generate training data
        noisy_energies, ideal_energies, observables, parameters = \
            self.extended_pipeline.generate_training_dataset(
                observable_list=observable_list,
                model_id=model_id
            )
        
        # Standard filename (no ID suffix)
        dataset_path = os.path.join(
            self.ml_dir, "train_data", f"train_data_{model_id}.csv"
        )
        logger.info(f"✓ Dataset saved: {dataset_path}")
        
        logger.info(f"\n[1.2] Training ML model...")
        logger.info(f"  - Model: RandomForest")
        logger.info(f"  - Trees: {n_estimators}")
        
        # Train model
        metrics = self.extended_pipeline.train_model(
            noisy_energies=noisy_energies,
            ideal_energies=ideal_energies,
            observables=observables,
            parameters=parameters,
            model_id=model_id,
            n_estimators=n_estimators
        )
        
        # Standard filename (no ID suffix)
        self.model_path = os.path.join(
            self.ml_dir, "ml_model", f"qem_model_{model_id}.pkl"
        )
        logger.info(f"✓ Model saved: {self.model_path}")
        
        # Log metrics
        logger.info(f"\n[1.3] Training Metrics:")
        for key, val in metrics.items():
            logger.info(f"  - {key}: {val:.6f}")
        
        self.model_metrics = metrics
        
        logger.info("\n✓ PHASE 1 COMPLETED\n")
        
        return {
            "status": "success",
            "experiment_id": self.experiment_id,
            "dataset_path": dataset_path,
            "model_path": self.model_path,
            "metrics": metrics
        }

    # ========================================================================
    # PHASE 2: ML EVALUATION
    # ========================================================================
    
    def phase2_ml_evaluation(
        self,
        test_data_path: str,
        model_id: int = 1
    ) -> Dict:
        """
        Phase 2: Evaluate trained model pada test data.
        
        Parameters
        ----------
        test_data_path : str
            Path ke test dataset CSV
        model_id : int
            Model identifier
            
        Returns
        -------
        Dict
            Evaluation results dengan predictions dan metrics
        """
        if self.model_path is None:
            logger.error("Model not trained! Run phase1_ml_training() first")
            raise ValueError("Model path not set. Run training phase first.")
        
        logger.info("=" * 70)
        logger.info("PHASE 2: ML EVALUATION")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info("=" * 70)
        
        # Load model
        logger.info(f"\n[2.1] Loading trained model...")
        logger.info(f"  - Path: {self.model_path}")
        self.extended_pipeline.ml_trainer.load_model(self.model_path)
        logger.info(f"✓ Model loaded")
        
        # Evaluate
        logger.info(f"\n[2.2] Evaluating on test data...")
        logger.info(f"  - Test data: {test_data_path}")
        
        test_results = self.extended_pipeline.evaluate_model(
            test_data_path=test_data_path,
            model_id=model_id
        )
        
        # Calculate metrics
        logger.info(f"\n[2.3] Evaluation Metrics:")
        ideal_col = 'ideal_energy'
        pred_col = 'RF_energy'
        
        if ideal_col in test_results.columns and pred_col in test_results.columns:
            from sklearn.metrics import mean_squared_error, r2_score
            
            mse = mean_squared_error(test_results[ideal_col], test_results[pred_col])
            r2 = r2_score(test_results[ideal_col], test_results[pred_col])
            mae = np.mean(np.abs(test_results[ideal_col] - test_results[pred_col]))
            
            logger.info(f"  - MSE: {mse:.6f}")
            logger.info(f"  - R²:  {r2:.4f}")
            logger.info(f"  - MAE: {mae:.6f}")
            
            eval_metrics = {"mse": mse, "r2": r2, "mae": mae}
        else:
            eval_metrics = {}
        
        # Standard filename (no ID suffix)
        pred_path = os.path.join(
            self.eval_dir, f"predictions_{model_id}.csv"
        )
        test_results.to_csv(pred_path, index=False)
        logger.info(f"✓ Predictions saved: {pred_path}")
        
        logger.info("\n✓ PHASE 2 COMPLETED\n")
        
        return {
            "status": "success",
            "experiment_id": self.experiment_id,
            "predictions_path": pred_path,
            "metrics": eval_metrics,
            "n_samples": len(test_results)
        }

    # ========================================================================
    # PHASE 3: VQE EXECUTION (dengan trained model)
    # ========================================================================
    
    def phase3_vqe_single_bond(
        self,
        bond_length: float,
        maxiter: int = 125
    ) -> Dict:
        """
        Phase 3a: VQE untuk single bond length dengan ML mitigation.
        
        Parameters
        ----------
        bond_length : float
            Bond length dalam Angstrom
        maxiter : int
            Max iterasi optimizer
            
        Returns
        -------
        Dict
            VQE results (ideal, noisy, zne, mitigated)
        """
        if self.model_path is None:
            logger.error("Model not trained! Run phase1_ml_training() first")
            raise ValueError("Model path not set. Run training phase first.")
        
        logger.info("=" * 70)
        logger.info(f"PHASE 3a: VQE EXECUTION (Single Bond Length: {bond_length})")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info("=" * 70)
        
        logger.info(f"\n[3a.1] Generating H2 Hamiltonian...")
        logger.info(f"  - Bond length: {bond_length} Å")
        
        # Generate Hamiltonian
        hamiltonian = self.vqe_pipeline.h2_gen.generate_h2_hamiltonian(bond_length)
        logger.info(f"✓ Hamiltonian generated")
        
        logger.info(f"\n[3a.2] Running VQE with ML-mitigated backend...")
        
        # Run VQE dengan model
        results = self.vqe_pipeline.run_full_vqe_pipeline(
            hamiltonian=hamiltonian,
            ml_model_path=self.model_path,
            maxiter=maxiter
        )
        
        logger.info(f"\n[3a.3] VQE Results:")
        logger.info(f"  - Ideal energy:    {results['ideal'].get('energy', 'N/A'):.6f}")
        logger.info(f"  - Noisy energy:    {results['noisy'].get('energy', 'N/A'):.6f}")
        logger.info(f"  - ZNE energy:      {results['zne'].get('energy', 'N/A'):.6f}")
        if results['mitigated'].get('status') != 'skipped':
            logger.info(f"  - Mitigated energy: {results['mitigated'].get('energy', 'N/A'):.6f}")
        
        logger.info("\n✓ PHASE 3a COMPLETED\n")
        
        return results

    def phase3_vqe_multi_bonds(
        self,
        bond_lengths: List[float],
        maxiter: int = 125
    ) -> Dict[float, Dict]:
        """
        Phase 3b: VQE untuk multiple bond lengths dengan ML mitigation.
        
        🎯 RECOMMENDED untuk Covalent integration
        
        Parameters
        ----------
        bond_lengths : List[float]
            Bond lengths dalam Angstrom
        maxiter : int
            Max iterasi optimizer
            
        Returns
        -------
        Dict[float, Dict]
            {bond_length: vqe_results}
        """
        if self.model_path is None:
            logger.error("Model not trained! Run phase1_ml_training() first")
            raise ValueError("Model path not set. Run training phase first.")
        
        logger.info("=" * 70)
        logger.info(f"PHASE 3b: VQE EXECUTION (Multiple Bond Lengths)")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info("=" * 70)
        logger.info(f"Bond lengths: {bond_lengths}")
        logger.info(f"ML model: {self.model_path}")
        
        # Run multi-bond VQE dengan model
        results = self.vqe_pipeline.run_full_vqe_pipeline_multi_bond_lengths(
            bond_lengths=bond_lengths,
            ml_model_path=self.model_path,
            maxiter=maxiter,
            save_results=True
        )
        
        logger.info("\n✓ PHASE 3b COMPLETED\n")
        
        return results

    # ========================================================================
    # COMPLETE WORKFLOW
    # ========================================================================
    
    def run_complete_workflow(
        self,
        observable_list: List[str],
        test_data_path: str,
        bond_lengths: List[float],
        model_id: int = 1,
        n_estimators: int = 100,
        maxiter: int = 125
    ) -> Dict:
        """
        Run complete end-to-end workflow:
        1. Train ML model
        2. Evaluate ML model
        3. Execute VQE dengan trained model
        
        Output organized dalam folder experiment_id!
        Filenames tetap standar (tanpa ID suffix).
        
        Parameters
        ----------
        observable_list : List[str]
            Training observables
        test_data_path : str
            Test data CSV path
        bond_lengths : List[float]
            Bond lengths untuk VQE
        model_id : int
            Model identifier
        n_estimators : int
            RF tree count
        maxiter : int
            VQE max iterations
            
        Returns
        -------
        Dict
            Complete workflow results
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 UNIFIED PIPELINE - COMPLETE WORKFLOW")
        logger.info(f"📌 Experiment ID: {self.experiment_id}")
        logger.info("=" * 70)
        
        workflow_results = {}
        
        # PHASE 1: Training
        try:
            phase1_results = self.phase1_ml_training(
                observable_list=observable_list,
                model_id=model_id,
                n_estimators=n_estimators
            )
            workflow_results['phase1_training'] = phase1_results
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            return {"status": "error", "phase": 1, "error": str(e)}
        
        # PHASE 2: Evaluation
        try:
            phase2_results = self.phase2_ml_evaluation(
                test_data_path=test_data_path,
                model_id=model_id
            )
            workflow_results['phase2_evaluation'] = phase2_results
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")
            return {"status": "error", "phase": 2, "error": str(e)}
        
        # PHASE 3: VQE
        try:
            phase3_results = self.phase3_vqe_multi_bonds(
                bond_lengths=bond_lengths,
                maxiter=maxiter
            )
            workflow_results['phase3_vqe'] = phase3_results
        except Exception as e:
            logger.error(f"Phase 3 failed: {e}")
            return {"status": "error", "phase": 3, "error": str(e)}
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("✅ WORKFLOW COMPLETE")
        logger.info(f"📌 Experiment ID: {self.experiment_id}")
        logger.info("=" * 70)
        logger.info(f"\nSummary:")
        logger.info(f"  Phase 1 - Training:    ✓ {phase1_results['status']}")
        logger.info(f"  Phase 2 - Evaluation:  ✓ {phase2_results['status']}")
        logger.info(f"  Phase 3 - VQE:         ✓ Complete ({len(bond_lengths)} bond lengths)")
        
        logger.info(f"\nOutput Directory:")
        logger.info(f"  {self.exp_dir}/")
        
        logger.info(f"\nOutput Files:")
        logger.info(f"  Training data: {phase1_results['dataset_path']}")
        logger.info(f"  ML model:      {phase1_results['model_path']}")
        logger.info(f"  Predictions:   {phase2_results['predictions_path']}")
        logger.info(f"  VQE results:   {self.vqe_dir}/vqe_results_multi_bond_length.csv")
        
        return workflow_results

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_workflow_summary(self) -> str:
        """Get summary dari completed workflow."""
        summary = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    UNIFIED PIPELINE WORKFLOW SUMMARY                      ║
║                                                                            ║
║  📌 Experiment ID: {self.experiment_id:<50} ║
║  📂 Output Dir:    {self.exp_dir:<50} ║
╚════════════════════════════════════════════════════════════════════════════╝

PHASE 1: ML TRAINING
  ├─ Generate training dataset (multiple observables)
  ├─ Train RandomForest model
  └─ Save qem_model_1.pkl (dalam folder experiment_id)

PHASE 2: ML EVALUATION
  ├─ Load test data
  ├─ Evaluate model accuracy
  └─ Calculate metrics (MSE, R², MAE)

PHASE 3: VQE EXECUTION
  ├─ Single bond length: run_phase3_vqe_single_bond()
  └─ Multiple bonds: run_phase3_vqe_multi_bonds() ⭐

All phases use the SAME trained ML model for error mitigation!

Output Files (Standard filenames, organized dalam folder experiment_id):
  ✓ train_data_1.csv
  ✓ qem_model_1.pkl
  ✓ predictions_1.csv
  ✓ vqe_results_multi_bond_length.csv
        """
        return summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    

    print("Custom experiment ID:")
    pipeline = UnifiedPipeline(
        n_qubits=4,
        n_circuits=100,
        experiment_id="h2_vqe_exp_001"
    )
    
    results = pipeline.run_complete_workflow(
    observable_list=["IIII", "IIIZ"],
    test_data_path="input/test_data/test_dataset.csv",
    bond_lengths=[0.735],
    maxiter=50
    )
    
