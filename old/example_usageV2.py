"""
example_usageV2.py - Contoh penggunaan ML-QEM Pipeline dengan qem_main_updated.py

File ini menunjukkan berbagai cara menggunakan pipeline:
1. Contoh 1: Simple usage (lokal, tanpa ML mitigation)
2. Contoh 2: Dengan ML model untuk mitigation
3. Contoh 3: Custom Hamiltonian (H2, LiH, etc)
4. Contoh 4: Covalent workflow untuk Quasi Brin
5. Contoh 5: Analisis dan visualisasi results

Setiap contoh bisa dijalankan standalone dengan `python example_usageV2.py --example N`
"""

import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

# Import dari qem_main_updated
from qem_main_updated import (
    QEMPipeline,
    VQERunner,
    GenerateAnsatz,
    AnsatzConfig,
    CalculateEnergy,
    TrainML
)

# Optional Covalent support
try:
    import covalent as ct
    COVALENT_AVAILABLE = True
except ImportError:
    COVALENT_AVAILABLE = False


# ============================================================================
# EXAMPLE 1: Simple Usage (Lokal, no ML)
# ============================================================================

def example_1_simple():
    """
    Contoh paling sederhana:
    - Hitung exact ground truth energy dengan NumPyMinimumEigensolver
    - Jalankan VQE noisy dengan SPSA
    - Bandingkan hasil
    
    Output: Ideal energy vs Noisy energy vs Error percentage
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Usage (Lokal, No ML)")
    print("="*70 + "\n")
    
    # Define simple Hamiltonian (H2-like)
    hamiltonian = SparsePauliOp(
        ["IIZZ", "IZZZ", "ZIZZ", "ZZIZ"],
        [0.1, 0.2, 0.15, 0.1]
    )
    
    # Create pipeline
    pipeline = QEMPipeline(n_qubits=4, output_dir="example1_output")
    
    # Run full VQE pipeline (ideal + noisy, no mitigation)
    results = pipeline.run_full_vqe_pipeline(
        hamiltonian=hamiltonian,
        ml_model_path=None,  # No ML model
        maxiter=50  # Cepat untuk testing
    )
    
    # Print hasil
    print("\nResults:")
    ideal_e = results["ideal"]["energy"]
    noisy_e = results["noisy"]["energy"]
    
    print(f"  Ideal energy (exact):     {ideal_e:.6f}")
    print(f"  Noisy energy (SPSA opt):  {noisy_e:.6f}")
    
    error_pct = abs(noisy_e - ideal_e) / abs(ideal_e) * 100
    print(f"  Error:                    {error_pct:.2f}%")
    
    print(f"\n✓ Results saved to: example1_output/")


# ============================================================================
# EXAMPLE 2: Dengan ML Mitigation
# ============================================================================

def example_2_with_ml():
    """
    Contoh dengan ML error mitigation:
    - Asumsikan sudah punya pre-trained RF model
    - Jalankan ideal + noisy + mitigated VQE
    - Compare ketiga hasil
    
    Requirement: qem_model.pkl sudah ada
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: With ML Mitigation")
    print("="*70 + "\n")
    
    hamiltonian = SparsePauliOp(
        ["IIZZ", "IZZZ", "ZIZZ", "ZZIZ"],
        [0.1, 0.2, 0.15, 0.1]
    )
    
    # Check apakah model file ada
    import os
    if not os.path.exists("qem_model.pkl"):
        print("⚠️  Warning: qem_model.pkl not found!")
        print("   Skipping ML mitigation example.")
        print("   Untuk run example ini, pastikan:")
        print("   1. qem_model.pkl sudah di-train")
        print("   2. File tersedia di current directory")
        return
    
    # Create pipeline
    pipeline = QEMPipeline(n_qubits=4, output_dir="example2_output")
    
    # Run dengan ML mitigation
    results = pipeline.run_full_vqe_pipeline(
        hamiltonian=hamiltonian,
        ml_model_path="qem_model.pkl",
        maxiter=50
    )
    
    # Print hasil
    print("\nResults with ML Mitigation:")
    ideal_e = results["ideal"]["energy"]
    noisy_e = results["noisy"]["energy"]
    
    if results["mitigated"].get("status") != "skipped":
        miti_e = results["mitigated"]["energy"]
        
        noisy_error = abs(noisy_e - ideal_e) / abs(ideal_e) * 100
        miti_error = abs(miti_e - ideal_e) / abs(ideal_e) * 100
        improvement = (noisy_error - miti_error) / noisy_error * 100
        
        print(f"  Ideal energy (exact):       {ideal_e:.6f}")
        print(f"  Noisy energy (SPSA):        {noisy_e:.6f} (error: {noisy_error:.2f}%)")
        print(f"  Mitigated energy (ML+SPSA): {miti_e:.6f} (error: {miti_error:.2f}%)")
        print(f"  Improvement:                {improvement:.1f}%")
    else:
        print(f"  ML mitigation skipped (model issue)")
    
    print(f"\n✓ Results saved to: example2_output/")


# ============================================================================
# EXAMPLE 3: Custom Hamiltonian
# ============================================================================

def example_3_custom_hamiltonian():
    """
    Contoh dengan custom Hamiltonian.
    
    Bisa gunakan:
    - H2 (4 qubits)
    - LiH (4 qubits)
    - Custom arbitrary Hamiltonian
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Hamiltonian")
    print("="*70 + "\n")
    
    # Define custom Hamiltonian
    print("Available Hamiltonians:")
    print("  1. H2-like (4 qubits)")
    print("  2. LiH-like (4 qubits)")
    print("  3. Custom (user-defined)")
    
    choice = input("\nSelect (1-3, default 1): ").strip() or "1"
    
    if choice == "1":
        # H2-like Hamiltonian
        hamiltonian = SparsePauliOp(
            ["IIZZ", "IZZZ", "ZIZZ", "ZZIZ"],
            [0.1, 0.2, 0.15, 0.1]
        )
        name = "H2-like"
    elif choice == "2":
        # LiH-like Hamiltonian (more complex)
        hamiltonian = SparsePauliOp(
            ["IIII", "IIZZ", "IZZZ", "ZIZZ", "ZZIZ", "ZZZI"],
            [0.05, 0.1, 0.15, 0.12, 0.08, 0.06]
        )
        name = "LiH-like"
    else:
        # Custom: User input
        labels = input("Enter Pauli labels (comma-separated, e.g., IIZZ,IZZZ): ").split(",")
        labels = [l.strip() for l in labels]
        coeffs = input("Enter coefficients (comma-separated, e.g., 0.1,0.2): ").split(",")
        coeffs = [float(c.strip()) for c in coeffs]
        hamiltonian = SparsePauliOp(labels, coeffs)
        name = "custom"
    
    print(f"\nUsing Hamiltonian: {name}")
    print(f"  Paulis: {[str(p) for p in hamiltonian.paulis]}")
    print(f"  Coeffs: {hamiltonian.coeffs}")
    
    # Run pipeline
    pipeline = QEMPipeline(n_qubits=4, output_dir=f"example3_{name}_output")
    
    results = pipeline.run_full_vqe_pipeline(
        hamiltonian=hamiltonian,
        maxiter=30  # Quick for demo
    )
    
    # Print results
    print(f"\nGround truth energy: {results['ideal']['energy']:.6f}")
    print(f"VQE (noisy) energy:  {results['noisy']['energy']:.6f}")
    
    print(f"\n✓ Results saved to: example3_{name}_output/")


# ============================================================================
# EXAMPLE 4: Covalent Workflow untuk Quasi Brin
# ============================================================================

def example_4_covalent_workflow():
    """
    Contoh Covalent workflow untuk remote execution di Quasi Brin.
    
    Workflow ini:
    1. Dispatch ke Covalent server
    2. Covalent forward ke SLURM di Quasi Brin
    3. Hasil di-retrieve via ct.get_result()
    
    Requirement: 
    - Covalent server harus running: `covalent start`
    - Atau server di Quasi Brin dengan proper address
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Covalent Workflow (Quasi Brin)")
    print("="*70 + "\n")
    
    if not COVALENT_AVAILABLE:
        print("⚠️  Covalent not installed!")
        print("   Install dengan: pip install covalent")
        return
    
    print("Covalent Workflow Modes:")
    print("  1. Local (Covalent server di laptop)")
    print("  2. Remote (Covalent server di Quasi Brin)")
    
    mode = input("\nSelect mode (1-2, default 1): ").strip() or "1"
    
    if mode == "1":
        # Local Covalent server
        print("\nSetup untuk Local mode:")
        print("  1. Start Covalent server: covalent start")
        print("  2. Monitor di: http://localhost:48008")
        print("  3. Code akan dispatch ke localhost:48008")
        
        # Define simple workflow
        from qem_main_updated import qem_vqe_workflow
        
        print("\nDispatching workflow to local Covalent server...")
        
        dispatch_id = ct.dispatch(qem_vqe_workflow)(
            n_qubits=4,
            hamiltonian_dict={
                "labels": ["IIZZ", "IZZZ"],
                "coeffs": [0.1, 0.2]
            },
            maxiter=50
        )
        
        print(f"✓ Workflow dispatched!")
        print(f"  Dispatch ID: {dispatch_id}")
        print(f"  Monitor at: http://localhost:48008")
        
        # Get result
        print("\nWaiting for results...")
        result = ct.get_result(dispatch_id, wait=True)
        
        print(f"\n✓ Workflow completed!")
        print(f"  Results: {result.result}")
    
    else:
        # Remote Covalent server (Quasi Brin)
        print("\nSetup untuk Remote mode (Quasi Brin):")
        print("  1. Setup Covalent server di Quasi Brin (see COVALENT_QUASI_BRIN_GUIDE.md)")
        print("  2. Set environment: export COVALENT_SERVER_ADDRESS=quasi-brin:48008")
        print("  3. Dispatch dari laptop tanpa perlu server lokal")
        
        print("\nNot implemented in this example.")
        print("Refer to: INTEGRATION_GUIDE.md untuk setup lengkap")


# ============================================================================
# EXAMPLE 5: Analisis dan Visualisasi Results
# ============================================================================

def example_5_analysis_visualization():
    """
    Contoh analisis dan visualisasi results dari VQE run.
    
    Baca JSON results file dan buat plot perbandingan.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Analysis & Visualization")
    print("="*70 + "\n")
    
    # Load results dari file
    import os
    
    results_file = "example1_output/vqe_results.json"
    
    if not os.path.exists(results_file):
        print(f"⚠️  Results file not found: {results_file}")
        print("   Run example 1 first: python example_usageV2.py --example 1")
        return
    
    # Load JSON
    with open(results_file) as f:
        results = json.load(f)
    
    print("Results summary:")
    print(json.dumps(results, indent=2))
    
    # Analysis
    if "ideal" in results and "noisy" in results:
        ideal_e = results["ideal"]["energy"]
        noisy_e = results["noisy"]["energy"]
        
        error = abs(noisy_e - ideal_e)
        error_pct = error / abs(ideal_e) * 100
        
        print(f"\nAnalysis:")
        print(f"  Ideal energy:  {ideal_e:.6f}")
        print(f"  Noisy energy:  {noisy_e:.6f}")
        print(f"  Absolute error: {error:.6f}")
        print(f"  Relative error: {error_pct:.2f}%")
        
        # Simple text plot
        print(f"\nEnergyComparison:")
        scale = 50
        ideal_bar = "█" * int(scale * abs(ideal_e) / max(abs(ideal_e), abs(noisy_e)))
        noisy_bar = "█" * int(scale * abs(noisy_e) / max(abs(ideal_e), abs(noisy_e)))
        
        print(f"  Ideal: {ideal_bar} {ideal_e:.4f}")
        print(f"  Noisy: {noisy_bar} {noisy_e:.4f}")


# ============================================================================
# EXAMPLE 6: Training ML Model
# ============================================================================

def example_6_train_ml_model():
    """
    Contoh training RandomForest model untuk ML mitigation.
    
    Steps:
    1. Generate training data (ideal + noisy energies)
    2. Extract features (noisy exp, params, gates, observables)
    3. Train RF model
    4. Save model
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Training ML Model")
    print("="*70 + "\n")
    
    print("Steps untuk train ML model:")
    print("  1. Generate training circuits (qem_main.py atau notebook)")
    print("  2. Calculate ideal & noisy energies")
    print("  3. Build feature matrix dari energies & observables")
    print("  4. Train RandomForestRegressor")
    print("  5. Save model dengan joblib")
    
    print("\nExample code:")
    print("""
from qem_main_updated import TrainML
import numpy as np
import pandas as pd

# Prepare training data
noisy_energies = np.array([...])  # dari SPSA optimization
ideal_energies = np.array([...])  # dari exact calculation
observables = ["IIZZ", "IZZZ", "ZIZZ", "ZZIZ"]
parameters = np.random.random(12)
gate_counts = {"x": 10, "sx": 15, "cx": 20}

# Build dataset
trainer = TrainML()
X, y = trainer.build_dataset(
    noisy_energies,
    ideal_energies,
    observables,
    parameters,
    gate_counts
)

# Train model
metrics = trainer.train_model(X, y, n_estimators=100)
print(f"Test R²: {metrics['test_r2']:.4f}")

# Save model
trainer.save_model("qem_model.pkl")

# Load model later
trainer.load_model("qem_model.pkl")
""")
    
    print("\nFor detailed example, see:")
    print("  - 02_ML_QEM_modeltrain.ipynb")
    print("  - INTEGRATION_GUIDE.md")


# ============================================================================
# EXAMPLE 7: Production Setup Mode B
# ============================================================================

def example_7_production_mode_b():
    """
    Contoh setup production Mode B:
    - Covalent server running di Quasi Brin (service/systemd)
    - Laptop hanya kirim workflow ke server
    - Laptop boleh ditutup setelah dispatch
    
    Workflow:
    1. Setup Covalent server di Quasi Brin
    2. Di laptop: Set COVALENT_SERVER_ADDRESS
    3. Dispatch workflow
    4. Monitor dari UI atau CLI
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Production Setup Mode B")
    print("="*70 + "\n")
    
    print("Mode B: Covalent Server di Quasi Brin (laptop boleh ditutup)")
    print("\nSetup steps:")
    print("""
1. SSH ke Quasi Brin dan install Covalent:
   ssh quasi-brin
   conda activate covalent_env
   pip install covalent

2. Start Covalent as service:
   covalent start --port 48008 --address 0.0.0.0
   (Atau setup systemd service, lihat: COVALENT_QUASI_BRIN_GUIDE.md)

3. Di laptop, set server address:
   export COVALENT_SERVER_ADDRESS=quasi-brin:48008

4. Dispatch workflow dari laptop:
   python example_usageV2.py --example 4

5. Laptop boleh ditutup, server akan terus running

6. Monitor progress:
   - Browser: http://quasi-brin:48008
   - CLI: covalent status <dispatch_id>
   - Python: ct.get_result(<dispatch_id>, wait=True)
""")
    
    print("\nBenefit Mode B:")
    print("  ✓ Laptop tidak perlu tetap hidup")
    print("  ✓ Server di resource-rich machine (Quasi Brin)")
    print("  ✓ Scalable ke multiple users")
    print("  ✓ Monitoring dari mana saja")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML-QEM Pipeline Usage Examples"
    )
    parser.add_argument(
        "--example", type=int, default=1,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Example number to run"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: example_1_simple,
        2: example_2_with_ml,
        3: example_3_custom_hamiltonian,
        4: example_4_covalent_workflow,
        5: example_5_analysis_visualization,
        6: example_6_train_ml_model,
        7: example_7_production_mode_b,
    }
    
    print(f"\nRunning Example {args.example}...")
    examples[args.example]()
    
    print("\n" + "="*70)
    print("Example completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
