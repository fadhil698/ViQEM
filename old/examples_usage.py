"""
Contoh Penggunaan ML-QEM (Machine Learning Quantum Error Mitigation)

File ini menunjukkan berbagai cara untuk menggunakan komponen-komponen
dalam program ML-QEM.
"""

# ============================================================================
# EXAMPLE 1: Penggunaan Full Pipeline (Cara Paling Sederhana)
# ============================================================================

def example_1_full_pipeline():
    """
    Menjalankan full pipeline dalam satu command
    """
    from old.qem_main import QEMPipeline
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Full Pipeline")
    print("="*70)
    
    # Buat pipeline dengan 4 qubit
    pipeline = QEMPipeline(n_qubits=4)
    
    # Define observables untuk diukur
    observables = [
        "ZZZZ",  # Semua Z
        "XXXX",  # Semua X
    ]
    
    # Jalankan full pipeline
    summary = pipeline.run_full_pipeline(
        n_circuits=20,          # Jumlah circuit
        observables=observables,
        n_estimators=100        # Jumlah trees RF
    )
    
    # Print hasil
    print("\nPipeline Summary:")
    print(f"  Qubits: {summary['n_qubits']}")
    print(f"  Circuits: {summary['n_circuits']}")
    print(f"  Observables: {summary['n_observables']}")
    print("\nModel Performance:")
    for key, value in summary['model_metrics'].items():
        print(f"  {key}: {value:.6f}")


# ============================================================================
# EXAMPLE 2: Step-by-Step dengan GenerateAnsatz
# ============================================================================

def example_2_generate_ansatz():
    """
    Membuat circuit ansatz dengan berbagai konfigurasi
    """
    from old.qem_main import GenerateAnsatz
    import numpy as np
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Generate Ansatz")
    print("="*70)
    
    # Konfigurasi 1: Ansatz dengan CX gate
    ansatz_gen_1 = GenerateAnsatz(
        n_qubits=4,
        entanglement_gate="cx",
        rotation_gate="ry",
        param_range=(-np.pi, np.pi),
        n_circuits=10
    )
    
    circuit_data_1 = ansatz_gen_1.create_circuit()
    print(f"\nAnsatz 1 Info:")
    print(f"  Jumlah circuit: {len(circuit_data_1.circuits)}")
    print(f"  Parameter shape: {circuit_data_1.param_values.shape}")
    print(f"  Config: {ansatz_gen_1.get_info()}")
    
    # Konfigurasi 2: Ansatz dengan CZ gate
    ansatz_gen_2 = GenerateAnsatz(
        n_qubits=4,
        entanglement_gate="cz",
        rotation_gate="rx",
        param_range=(-np.pi/2, np.pi/2),
        n_circuits=15
    )
    
    circuit_data_2 = ansatz_gen_2.create_circuit()
    print(f"\nAnsatz 2 Info:")
    print(f"  Jumlah circuit: {len(circuit_data_2.circuits)}")
    print(f"  Parameter shape: {circuit_data_2.param_values.shape}")
    print(f"  Config: {ansatz_gen_2.get_info()}")
    
    # Visualisasi circuit pertama
    print("\nVisualisasi Circuit Pertama (Ansatz 1):")
    print(circuit_data_1.circuits[0].draw())
    
    return circuit_data_1


# ============================================================================
# EXAMPLE 3: Calculate Energies
# ============================================================================

def example_3_calculate_energies(circuit_data):
    """
    Menghitung ideal, noisy, dan ZNE energies
    """
    from old.qem_main import CalculateEnergy
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Calculate Energies")
    print("="*70)
    
    # Setup observable
    observables = ["ZZZZ", "XXXX", "XYXY"]
    
    # Buat energy calculator
    energy_calc = CalculateEnergy(
        circuits=circuit_data.circuits,
        observables=observables,
        backend_name="FakeAthensV2"
    )
    
    # Hitung energies untuk semua circuit dan observable
    energy_result = energy_calc.calculate_all_energies()
    
    print(f"\nEnergy Calculation Results:")
    print(f"  Ideal Energy Shape: {energy_result.ideal_energy.shape}")
    print(f"  Noisy Energy Shape: {energy_result.noisy_energy.shape}")
    print(f"  ZNE Energy Shape: {energy_result.zne_energy.shape}")
    print(f"  Observables: {energy_result.observables}")
    
    # Show beberapa sample
    print(f"\nSample Energy Values (Circuit 0, Observable ZZZZ):")
    print(f"  Ideal: {energy_result.ideal_energy[0, 0]:.6f}")
    print(f"  Noisy: {energy_result.noisy_energy[0, 0]:.6f}")
    print(f"  ZNE:   {energy_result.zne_energy[0, 0]:.6f}")
    
    return energy_result


# ============================================================================
# EXAMPLE 4: Train ML Model
# ============================================================================

def example_4_train_ml(energy_result):
    """
    Melatih Random Forest model untuk error mitigation
    """
    from old.qem_main import TrainML
    import os
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Train ML Model")
    print("="*70)
    
    # Create trainer
    ml_trainer = TrainML(
        ideal_energy=energy_result.ideal_energy,
        noisy_energy=energy_result.noisy_energy,
        zne_energy=energy_result.zne_energy,
        observables=energy_result.observables,
        test_size=0.2,
        random_state=42
    )
    
    # Encode observables
    observable_encoding = ml_trainer.encode_observables()
    print(f"\nObservable Encoding:")
    for obs, code in observable_encoding.items():
        print(f"  {obs}: {code}")
    
    # Build dataset
    X, y = ml_trainer.build_dataset()
    print(f"\nDataset Info:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    # Train model
    metrics = ml_trainer.train_model(
        n_estimators=150,
        max_depth=15,
        verbose=1
    )
    
    print(f"\nModel Performance Metrics:")
    print(f"  Train MSE: {metrics['train_mse']:.6f}")
    print(f"  Test MSE:  {metrics['test_mse']:.6f}")
    print(f"  Train MAE: {metrics['train_mae']:.6f}")
    print(f"  Test MAE:  {metrics['test_mae']:.6f}")
    print(f"  Train R²:  {metrics['train_r2']:.6f}")
    print(f"  Test R²:   {metrics['test_r2']:.6f}")
    
    # Save model
    model_path = "qem_model_example.pkl"
    ml_trainer.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Feature importance
    feature_importance = ml_trainer.model.feature_importances_
    print(f"\nFeature Importance:")
    print(f"  Noisy Energy:     {feature_importance[0]:.6f}")
    print(f"  ZNE Energy:       {feature_importance[1]:.6f}")
    print(f"  Observable Code:  {feature_importance[2]:.6f}")
    
    return ml_trainer


# ============================================================================
# EXAMPLE 5: Make Predictions
# ============================================================================

def example_5_predictions(ml_trainer):
    """
    Membuat prediksi dengan trained model
    """
    import numpy as np
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Make Predictions")
    print("="*70)
    
    # Test data (simulasi)
    test_noisy_energy = np.array([[0.5, 1.2, 0.8]])
    test_zne_energy = np.array([[0.7, 1.4, 0.9]])
    observable_code = np.array([[0]])
    
    # Combine features
    test_X = np.hstack([
        test_noisy_energy,
        test_zne_energy,
        observable_code
    ])
    
    # Predict
    predicted_ideal = ml_trainer.predict(test_X)
    
    print(f"\nSample Prediction:")
    print(f"  Noisy Energy Input: {test_noisy_energy[0, 0]:.6f}")
    print(f"  ZNE Energy Input:   {test_zne_energy[0, 0]:.6f}")
    print(f"  Predicted Ideal:    {predicted_ideal[0]:.6f}")


# ============================================================================
# EXAMPLE 6: Custom Configuration
# ============================================================================

def example_6_custom_config():
    """
    Menjalankan pipeline dengan konfigurasi custom
    """
    from old.qem_main import QEMPipeline
    import numpy as np
    
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Configuration")
    print("="*70)
    
    # Konfigurasi custom
    config = {
        "n_qubits": 3,
        "n_circuits": 25,
        "entanglement_gate": "cz",
        "rotation_gate": "rx",
        "n_estimators": 200,
        "max_depth": 20
    }
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create pipeline
    pipeline = QEMPipeline(n_qubits=config["n_qubits"])
    
    # Generate ansatz
    circuit_data = pipeline.generate_ansatz(
        entanglement_gate=config["entanglement_gate"],
        rotation_gate=config["rotation_gate"],
        n_circuits=config["n_circuits"]
    )
    
    # Custom observables
    observables = [
        "ZZZ",
        "XXX",
        "YYY",
        "ZXZ",
        "XZX"
    ]
    
    # Calculate energies
    energy_result = pipeline.calculate_energies(
        circuits=circuit_data.circuits,
        observables=observables
    )
    
    # Train model
    metrics = pipeline.train_mitigation_model(
        ideal_energy=energy_result.ideal_energy,
        noisy_energy=energy_result.noisy_energy,
        zne_energy=energy_result.zne_energy,
        observables=energy_result.observables,
        n_estimators=config["n_estimators"]
    )
    
    print(f"\nResults:")
    print(f"  Test MSE: {metrics['test_mse']:.6f}")
    print(f"  Test R²:  {metrics['test_r2']:.6f}")


# ============================================================================
# EXAMPLE 7: Load and Use Saved Model
# ============================================================================

def example_7_load_saved_model():
    """
    Load model yang sudah disimpan dan gunakan untuk prediksi baru
    """
    from old.qem_main import TrainML
    import numpy as np
    import os
    
    print("\n" + "="*70)
    print("EXAMPLE 7: Load Saved Model")
    print("="*70)
    
    model_path = "qem_model_example.pkl"
    
    if os.path.exists(model_path):
        # Load model
        ml_trainer = TrainML(
            ideal_energy=np.zeros((1, 1)),  # Dummy data
            noisy_energy=np.zeros((1, 1)),
            zne_energy=np.zeros((1, 1)),
            observables=["Z"]
        )
        ml_trainer.load_model(model_path)
        print(f"\nModel loaded from: {model_path}")
        
        # Use model
        test_input = np.array([[0.3, 0.5, 1]])
        prediction = ml_trainer.predict(test_input)
        print(f"Prediction: {prediction[0]:.6f}")
    else:
        print(f"\nModel file not found: {model_path}")
        print("Jalankan example_4_train_ml() terlebih dahulu")


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + " " * 20 + "ML-QEM USAGE EXAMPLES" + " " * 28 + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    try:
        # Run examples
        # example_1_full_pipeline()
        
        circuit_data = example_2_generate_ansatz()
        # energy_result = example_3_calculate_energies(circuit_data)
        # ml_trainer = example_4_train_ml(energy_result)
        # example_5_predictions(ml_trainer)
        
        example_6_custom_config()
        # example_7_load_saved_model()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
