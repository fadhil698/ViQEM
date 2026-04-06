# ============================================================================
# UPDATE: VQEPipeline dengan Support Multiple Bond Lengths
# ============================================================================

from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
import logging
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)

# ============================================================================
# CLASS: VQEPipeline (UPDATED)
# ============================================================================

class VQEPipeline:
    """
    Orchestrate full ML-QEM VQE pipeline dengan support untuk MULTIPLE BOND LENGTHS.
    
    Update:
    - Tambah method: run_full_vqe_pipeline_multi_bond_lengths()
    - Support input: List[float] bond_lengths
    - Output: Dict[float, Dict] hasil untuk setiap bond length
    """

    def __init__(
        self,
        n_qubits: int = 4,
        output_dir: str = "qem_output"
    ):
        from ML_QEM.qem_main_updated_v2 import (
            GenerateAnsatz,
            AnsatzConfig,
            CalculateEnergy,
            VQERunner,
            TrainML,
            H2HamiltonianGenerator
        )
        
        self.n_qubits = n_qubits
        self.output_dir = output_dir
        
        config = AnsatzConfig(n_qubits=n_qubits)
        self.ansatz_gen = GenerateAnsatz(config)
        self.energy_calc = CalculateEnergy(n_qubits)
        self.ml_trainer = TrainML()
        self.h2_gen = H2HamiltonianGenerator(n_qubits=n_qubits)
        
        logger.info(f"VQEPipeline initialized: {n_qubits} qubits (UPDATED with multi-bond-length support)")

    def run_full_vqe_pipeline(
        self,
        hamiltonian: SparsePauliOp,
        ml_model_path: Optional[str] = None,
        maxiter: int = 125
    ) -> Dict:
        """
        Run full VQE pipeline untuk SATU Hamiltonian (kompatibel dengan original).
        
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
            Hasil VQE (ideal, noisy, ZNE, mitigated)
        """
        logger.info("=" * 70)
        logger.info("QEM FULL VQE PIPELINE (SINGLE BOND LENGTH)")
        logger.info("=" * 70)

        from ML_QEM.qem_main_updated_v2 import VQERunner
        
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

    # ========================================================================
    # NEW METHOD: Support Multiple Bond Lengths
    # ========================================================================
    
    def run_full_vqe_pipeline_multi_bond_lengths(
        self,
        bond_lengths: List[float],
        ml_model_path: Optional[str] = None,
        maxiter: int = 125,
        save_results: bool = True
    ) -> Dict[float, Dict]:
        """
        Run full VQE pipeline untuk MULTIPLE bond lengths.
        
        FITUR BARU:
        - Input: List[float] bond_lengths (misal [0.5, 0.6, 0.735, 1.0, 1.5])
        - Secara otomatis generate Hamiltonian untuk setiap bond length
        - Jalankan full VQE pipeline (ideal, noisy, ZNE, mitigated) untuk masing-masing
        - Return struktur Dict[bond_length] untuk mudah diakses
        
        Parameters
        ----------
        bond_lengths : List[float]
            List of bond lengths dalam Angstrom (misal [0.5, 0.6, 0.735, 1.0])
        ml_model_path : Optional[str]
            Path ke ML model untuk error mitigation
        maxiter : int
            Max iterasi optimizer (default 125)
        save_results : bool
            Jika True, simpan hasil ke CSV per bond length
            
        Returns
        -------
        Dict[float, Dict]
            Struktur: {
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
        
        from ML_QEM.qem_main_updated_v2 import VQERunner
        
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
                    'mitigated': f"{res['mitigated'].get('energy', np.nan) if res['mitigated'].get('status') != 'skipped' else 'SKIPPED':.6f}" if res['mitigated'].get('status') != 'skipped' else "SKIPPED"
                })
        
        summary_df = pd.DataFrame(summary_rows)
        logger.info("\n" + summary_df.to_string(index=False))
        
        return results_all
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _save_results(self, results: Dict):
        """Save results untuk single bond length ke CSV (original)."""
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

    def _save_multi_bond_length_results(self, results_all: Dict[float, Dict]):
        """
        Save results untuk multiple bond lengths ke CSV.
        
        Struktur output:
        bond_length, ideal_energy, noisy_energy, zne_energy, mitigated_energy, status
        0.5000,      -1.123456,   -1.098765,  -1.115432,  -1.112345,        done
        0.6000,      -1.234567,   -1.203456,  -1.225678,  -1.223456,        done
        ...
        """
        import os
        
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
        
        # Create output directory jika belum ada
        output_dir = "qem_output"
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, "vqe_results_multi_bond_length.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Multi-bond-length results saved to: {csv_path}")
        
        # Print tabel
        logger.info("\nResults Table:")
        logger.info(df.to_string(index=False))

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    )
    
    # Initialize pipeline
    pipeline = VQEPipeline(n_qubits=4)
    
    # Define bond lengths
    bond_lengths = [0.5, 0.6, 0.735, 1.0, 1.5]
    
    # Run VQE untuk multiple bond lengths
    results = pipeline.run_full_vqe_pipeline_multi_bond_lengths(
        bond_lengths=bond_lengths,
        ml_model_path="output/ml_model/qem_model_1.pkl",  # Optional
        maxiter=125,
        save_results=True
    )
    
    # Access hasil
    print("\n" + "="*70)
    print("ACCESSING RESULTS")
    print("="*70)
    
    for bl in [0.5, 0.735, 1.5]:
        if bl in results:
            print(f"\nBond length {bl}:")
            print(f"  Ideal:    {results[bl]['ideal']['energy']:.6f}")
            print(f"  Noisy:    {results[bl]['noisy']['energy']:.6f}")
            print(f"  ZNE:      {results[bl]['zne']['energy']:.6f}")
            if results[bl]['mitigated'].get('status') != 'skipped':
                print(f"  Mitigated: {results[bl]['mitigated']['energy']:.6f}")
