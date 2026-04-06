"""
Konfigurasi Default untuk ML-QEM

File ini berisi konfigurasi yang dapat dengan mudah disesuaikan
untuk berbagai skenario quantum computing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# ============================================================================
# KONFIGURASI ANSATZ
# ============================================================================

ANSATZ_CONFIG = {
    "default": {
        "n_qubits": 4,
        "entanglement_gate": "cx",
        "rotation_gate": "ry",
        "param_range": (-np.pi, np.pi),
        "n_circuits": 50,
        "description": "Default configuration dengan 4 qubits dan CX entanglement"
    },
    
    "small_system": {
        "n_qubits": 2,
        "entanglement_gate": "cx",
        "rotation_gate": "ry",
        "param_range": (-np.pi, np.pi),
        "n_circuits": 20,
        "description": "Konfigurasi untuk sistem kecil (2 qubits)"
    },
    
    "medium_system": {
        "n_qubits": 6,
        "entanglement_gate": "cx",
        "rotation_gate": "ry",
        "param_range": (-np.pi, np.pi),
        "n_circuits": 100,
        "description": "Konfigurasi untuk sistem medium (6 qubits)"
    },
    
    "large_system": {
        "n_qubits": 10,
        "entanglement_gate": "cx",
        "rotation_gate": "ry",
        "param_range": (-np.pi, np.pi),
        "n_circuits": 200,
        "description": "Konfigurasi untuk sistem besar (10 qubits)"
    },
    
    "high_precision": {
        "n_qubits": 4,
        "entanglement_gate": "cx",
        "rotation_gate": "rx",
        "param_range": (-np.pi/4, np.pi/4),
        "n_circuits": 150,
        "description": "Konfigurasi untuk high-precision calculations"
    },
    
    "vqe_style": {
        "n_qubits": 4,
        "entanglement_gate": "cx",
        "rotation_gate": "ry",
        "param_range": (-2*np.pi, 2*np.pi),
        "n_circuits": 100,
        "description": "Konfigurasi style VQE dengan parameter range lebih luas"
    }
}

# ============================================================================
# KONFIGURASI OBSERVABLES
# ============================================================================

OBSERVABLE_SETS = {
    "paulix": {
        "observables": ["XXXX", "XXXX"],
        "description": "Semua Pauli X measurements"
    },
    
    "pauliz": {
        "observables": ["ZZZZ", "ZZZZ"],
        "description": "Semua Pauli Z measurements"
    },
    
    "pauliy": {
        "observables": ["YYYY", "YYYY"],
        "description": "Semua Pauli Y measurements"
    },
    
    "mixed_paulis": {
        "observables": ["ZZZZ", "XXXX", "YYYY"],
        "description": "Kombinasi Pauli X, Y, Z"
    },
    
    "alternating": {
        "observables": ["ZXZX", "XZXZ", "ZYZY", "YZYZ"],
        "description": "Alternating Pauli measurements"
    },
    
    "hamiltonian": {
        "observables": ["ZZZZ", "XXXX", "YXYX", "ZYZY"],
        "description": "Observable set untuk Hamiltonian simulation"
    },
    
    "comprehensive": {
        "observables": [
            "ZZZZ", "XXXX", "YYYY",
            "ZXZX", "XZXZ", "ZYZY",
            "YZYZ", "ZXXX", "XZXX"
        ],
        "description": "Comprehensive observable set untuk testing"
    }
}

# ============================================================================
# KONFIGURASI BACKEND
# ============================================================================

BACKEND_CONFIG = {
    "fake_athens": {
        "backend_name": "FakeAthensV2",
        "use_real_backend": False,
        "description": "IBM FakeAthensV2 (5 qubits simulator)"
    },
    
    "statevector": {
        "backend_name": "AerSimulator",
        "method": "statevector",
        "use_real_backend": False,
        "description": "Qiskit Aer Statevector Simulator"
    },
    
    "unitary": {
        "backend_name": "AerSimulator",
        "method": "unitary",
        "use_real_backend": False,
        "description": "Qiskit Aer Unitary Simulator"
    }
}

# ============================================================================
# KONFIGURASI MACHINE LEARNING MODEL
# ============================================================================

ML_MODEL_CONFIG = {
    "light": {
        "model_type": "RandomForestRegressor",
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "test_size": 0.2,
        "random_state": 42,
        "description": "Light model untuk quick testing"
    },
    
    "default": {
        "model_type": "RandomForestRegressor",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "test_size": 0.2,
        "random_state": 42,
        "description": "Default RandomForest configuration"
    },
    
    "medium": {
        "model_type": "RandomForestRegressor",
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "test_size": 0.2,
        "random_state": 42,
        "description": "Medium complexity model"
    },
    
    "heavy": {
        "model_type": "RandomForestRegressor",
        "n_estimators": 500,
        "max_depth": None,  # Unlimited depth
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "test_size": 0.1,
        "random_state": 42,
        "description": "Heavy model untuk maximum accuracy"
    },
    
    "production": {
        "model_type": "RandomForestRegressor",
        "n_estimators": 300,
        "max_depth": 25,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "test_size": 0.15,
        "random_state": 42,
        "description": "Production-ready configuration"
    }
}

# ============================================================================
# PRESET PIPELINES
# ============================================================================

PIPELINE_PRESETS = {
    "quick_test": {
        "ansatz_config": ANSATZ_CONFIG["small_system"],
        "observable_set": OBSERVABLE_SETS["pauliz"],
        "backend_config": BACKEND_CONFIG["fake_athens"],
        "ml_config": ML_MODEL_CONFIG["light"],
        "description": "Quick test pipeline (small & fast)"
    },
    
    "standard": {
        "ansatz_config": ANSATZ_CONFIG["default"],
        "observable_set": OBSERVABLE_SETS["mixed_paulis"],
        "backend_config": BACKEND_CONFIG["fake_athens"],
        "ml_config": ML_MODEL_CONFIG["default"],
        "description": "Standard pipeline untuk testing umum"
    },
    
    "research": {
        "ansatz_config": ANSATZ_CONFIG["medium_system"],
        "observable_set": OBSERVABLE_SETS["comprehensive"],
        "backend_config": BACKEND_CONFIG["fake_athens"],
        "ml_config": ML_MODEL_CONFIG["medium"],
        "description": "Research-grade pipeline"
    },
    
    "production": {
        "ansatz_config": ANSATZ_CONFIG["large_system"],
        "observable_set": OBSERVABLE_SETS["comprehensive"],
        "backend_config": BACKEND_CONFIG["fake_athens"],
        "ml_config": ML_MODEL_CONFIG["production"],
        "description": "Production-ready pipeline"
    },
    
    "high_precision": {
        "ansatz_config": ANSATZ_CONFIG["high_precision"],
        "observable_set": OBSERVABLE_SETS["hamiltonian"],
        "backend_config": BACKEND_CONFIG["fake_athens"],
        "ml_config": ML_MODEL_CONFIG["heavy"],
        "description": "High precision pipeline"
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_ansatz_config(config_name: str = "default") -> Dict:
    """
    Dapatkan konfigurasi ansatz
    
    Parameters
    ----------
    config_name : str
        Nama konfigurasi preset
        
    Returns
    -------
    Dict
        Konfigurasi ansatz
    """
    if config_name not in ANSATZ_CONFIG:
        print(f"Available configs: {list(ANSATZ_CONFIG.keys())}")
        config_name = "default"
    
    return ANSATZ_CONFIG[config_name].copy()


def get_observable_set(set_name: str = "mixed_paulis") -> List[str]:
    """
    Dapatkan set observables
    
    Parameters
    ----------
    set_name : str
        Nama observable set preset
        
    Returns
    -------
    List[str]
        List observable strings
    """
    if set_name not in OBSERVABLE_SETS:
        print(f"Available sets: {list(OBSERVABLE_SETS.keys())}")
        set_name = "mixed_paulis"
    
    return OBSERVABLE_SETS[set_name]["observables"]


def get_ml_config(config_name: str = "default") -> Dict:
    """
    Dapatkan konfigurasi ML model
    
    Parameters
    ----------
    config_name : str
        Nama ML config preset
        
    Returns
    -------
    Dict
        Konfigurasi ML model
    """
    if config_name not in ML_MODEL_CONFIG:
        print(f"Available configs: {list(ML_MODEL_CONFIG.keys())}")
        config_name = "default"
    
    return ML_MODEL_CONFIG[config_name].copy()


def get_pipeline_preset(preset_name: str = "standard") -> Dict:
    """
    Dapatkan preset pipeline lengkap
    
    Parameters
    ----------
    preset_name : str
        Nama pipeline preset
        
    Returns
    -------
    Dict
        Konfigurasi pipeline lengkap
    """
    if preset_name not in PIPELINE_PRESETS:
        print(f"Available presets: {list(PIPELINE_PRESETS.keys())}")
        preset_name = "standard"
    
    return PIPELINE_PRESETS[preset_name].copy()


def print_available_configs():
    """Print semua konfigurasi yang tersedia"""
    print("\n" + "="*70)
    print("AVAILABLE CONFIGURATIONS")
    print("="*70)
    
    print("\nANSATZ CONFIGURATIONS:")
    for name, config in ANSATZ_CONFIG.items():
        print(f"  {name:15} - {config['description']}")
    
    print("\nOBSERVABLE SETS:")
    for name, obs_set in OBSERVABLE_SETS.items():
        print(f"  {name:15} - {obs_set['description']}")
    
    print("\nML MODEL CONFIGURATIONS:")
    for name, config in ML_MODEL_CONFIG.items():
        print(f"  {name:15} - {config['description']}")
    
    print("\nPIPELINE PRESETS:")
    for name, preset in PIPELINE_PRESETS.items():
        print(f"  {name:15} - {preset['description']}")
    
    print("\n" + "="*70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Print semua konfigurasi
    print_available_configs()
    
    # Contoh menggunakan konfigurasi
    print("\n" + "="*70)
    print("EXAMPLE USAGE")
    print("="*70)
    
    # Dapatkan preset pipeline
    preset = get_pipeline_preset("quick_test")
    print(f"\nQuick Test Preset:")
    print(f"  Qubits: {preset['ansatz_config']['n_qubits']}")
    print(f"  Circuits: {preset['ansatz_config']['n_circuits']}")
    print(f"  ML Model: {preset['ml_config']['model_type']}")
    print(f"  N Estimators: {preset['ml_config']['n_estimators']}")
    
    # Dapatkan konfigurasi individual
    ansatz_config = get_ansatz_config("medium_system")
    print(f"\nMedium System Ansatz:")
    print(f"  N Qubits: {ansatz_config['n_qubits']}")
    print(f"  Entanglement: {ansatz_config['entanglement_gate']}")
    print(f"  Param Range: {ansatz_config['param_range']}")
