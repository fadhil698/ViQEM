import numpy as np
import pandas as pd
import os
import time
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit_aer.noise import NoiseModel



from ML_QEM.qem_main_updated_v2 import (
    GenerateAnsatz, AnsatzConfig, CalculateEnergy
)
from custom_noise import CustomNoiseBuilder

# ============================================================================
# CONFIGURATION
# ============================================================================
N_CIRCUITS = 2000
DATA_CONDITION = "custom7"
BASE_OUTPUT_PATH = "input/test_data"
target_qubits = [2, 3]
MAX_WORKERS = 8  # Sesuaikan dengan jumlah core CPU Anda

ALL_OBSERVABLES = {
    "jw": ['IIII', 'IIIZ', 'IIZI', 'IIZZ', 'IZII', 'IZIZ', 'ZIII', 'ZIIZ', 'YYYY', 'XXYY', 'YYXX', 'XXXX', 'IZZI', 'ZIZI', 'ZZII'],
    "bk": ['IIII', 'IIIZ', 'IIZZ', 'IIZI', 'IZII', 'IZIZ', 'ZZZI', 'ZZZZ', 'ZXIX', 'IXZX', 'ZXZX', 'IXIX', 'IZZZ', 'ZZIZ', 'ZIZI'],
    "parity": ['II', 'IZ', 'ZI', 'ZZ', 'XX']
}

# ============================================================================
# WORKER FUNCTION
# ============================================================================

def generate_single_circuit_data(idx: int, mapper: str, n_obs: int, obs_list: List[str], 
                                 n_qubits: int, n_params: int, noise_model: Any) -> Dict[str, Any]:
    """
    Fungsi ini dijalankan di dalam worker process untuk menghitung satu baris data.
    """
    # Import lokal di dalam worker jika diperlukan oleh environment tertentu
    from qiskit.quantum_info import SparsePauliOp
    
    # Re-inisialisasi calculator di dalam worker (lebih aman untuk multiprocessing)
    # Gunakan template ansatz sederhana di dalam worker atau kirimkan via argumen
    # Namun untuk efisiensi, kita hitung energinya di sini
    
    # Setup calculator
    energy_calc = CalculateEnergy(n_qubits=n_qubits, noise_model=noise_model)
    
    # Pilih observable & parameter
    pauli = obs_list[idx % n_obs]
    obs_op = SparsePauliOp(pauli)
    theta = np.random.uniform(-np.pi, np.pi, size=n_params)
    
    # Kita perlu membuat ansatz lokal di worker karena Qiskit objects kadang sulit di-pickle
    # (Opsional: Jika GenerateAnsatz berat, sebaiknya buat sirkuit sekali saja)
    config = AnsatzConfig(n_qubits=n_qubits, reps=2)
    gen = GenerateAnsatz(config)
    ansatz = gen.create_ansatz()

    # Hitung Energy
    ideal_e = energy_calc.calculate_ideal_energy(ansatz, obs_op, theta)
    noisy_e = energy_calc.calculate_noisy_energy(ansatz, obs_op, theta)
    zne_e = energy_calc.calculate_zne_energy(ansatz, obs_op, theta)

    # Susun baris data
    result = {
        "noisy_energy": float(noisy_e),
        "ideal_energy": float(ideal_e),
        "zne_energy": float(zne_e),
        "observable": pauli,
    }
    for j, p in enumerate(theta):
        result[f"param_{j}"] = p
        
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    start_total = time.time()
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    
    # Noise model di-build sekali dan dikirim ke workers
    builder = CustomNoiseBuilder()
    noise_model = builder.get_noise_model(target_qubits=target_qubits)


    # backend_fake = FakeBelemV2()
    # noise_model = NoiseModel.from_backend(backend_fake)
    
    for mapper, obs_list in ALL_OBSERVABLES.items():
        n_qubits = 2 if mapper == "parity" else 4
        n_obs = len(obs_list)
        
        # Hitung jumlah parameter sekali untuk template
        temp_config = AnsatzConfig(n_qubits=n_qubits, reps=2)
        n_params = GenerateAnsatz(temp_config).create_ansatz().num_parameters

        print(f"\n🚀 Processing {mapper.upper()} with {MAX_WORKERS} workers...")
        
        results = []
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit semua task sirkuit ke pool
            futures = [
                executor.submit(
                    generate_single_circuit_data, 
                    i, mapper, n_obs, obs_list, n_qubits, n_params, noise_model
                ) for i in range(N_CIRCUITS)
            ]
            
            # Ambil hasil saat selesai
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % 200 == 0:
                    print(f"  > {mapper.upper()}: {completed}/{N_CIRCUITS} done")

        # Simpan ke DataFrame
        df = pd.DataFrame(results)
        csv_path = os.path.join(BASE_OUTPUT_PATH, f"test_data_{mapper}_{DATA_CONDITION}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Finished {mapper.upper()}. Saved to: {csv_path}")

    print(f"\n{'='*60}")
    print(f"🏁 ALL DATASETS COMPLETED in {time.time() - start_total:.2f} seconds")
    print(f"{'='*60}")