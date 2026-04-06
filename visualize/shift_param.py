import numpy as np
import pandas as pd
from qiskit.circuit.library import n_local
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------
# 1. KONFIGURASI GLOBAL
# ---------------------------------------------------------
def create_ansatz():
    return n_local(
        num_qubits=4,
        rotation_blocks='ry',
        entanglement_blocks='cx',
        entanglement='linear',
        reps=2,
        insert_barriers=True
    )

def compute_worker(val, obs_str, ds_name):
    """
    Fungsi ini akan dijalankan di setiap core CPU secara terpisah.
    """
    ansatz = create_ansatz()
    estimator = StatevectorEstimator()
    hamiltonian = SparsePauliOp.from_list([(obs_str, 1.0)])
    
    shift = 0.5
    current_params = np.array([val] * 12)
    
    # 1. Hitung Energi Pusat
    pub_center = (ansatz, hamiltonian, current_params)
    e_center = float(estimator.run([pub_center]).result()[0].data.evs)
    
    entry = {
        'dataset': ds_name,
        'observable': obs_str,
        'input_val': val,
        'ideal_energy': e_center
    }
    
    # 2. Hitung Gradien (Parameter Shift)
    for i in range(12):
        p_plus = current_params.copy()
        p_minus = current_params.copy()
        p_plus[i] += shift
        p_minus[i] -= shift
        
        pub_plus = (ansatz, hamiltonian, p_plus)
        pub_minus = (ansatz, hamiltonian, p_minus)
        
        # Jalankan sekaligus untuk efisiensi
        results = estimator.run([pub_plus, pub_minus]).result()
        e_plus = float(results[0].data.evs)
        e_minus = float(results[1].data.evs)
        
        entry[f'grad_param_{i}'] = (e_plus - e_minus) / (2 * shift)
        
    return entry

# ---------------------------------------------------------
# 2. EKSEKUSI UTAMA
# ---------------------------------------------------------
if __name__ == '__main__':
    df_input = pd.read_csv('visualize/all_observable.csv')

    dataset_mapping = {k: v for k, v in df_input.groupby('dataset')['observable'].first().to_dict().items() 
                       if k.startswith('single')}

    # UBAH STEP DI SINI: 0.1 agar lebih cepat, atau 0.005 untuk data produksi
    param_values = np.arange(-5, 5.1, 0.1) 
    
    
    all_tasks = []
    for ds_name, obs_str in dataset_mapping.items():
        for val in param_values:
            all_tasks.append((val, obs_str, ds_name))

    results_list = []
    print(f"Memulai komputasi paralel untuk {len(all_tasks)} total titik...")

    # Menggunakan ProcessPoolExecutor untuk membagi tugas ke semua core CPU
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_worker, *task) for task in all_tasks]
        
        # Progress bar untuk memantau status secara real-time
        for fut in tqdm(as_completed(futures), total=len(all_tasks), desc="Total Progress"):
            results_list.append(fut.result())

    # Simpan Hasil
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(['dataset', 'input_val']) # Urutkan agar rapi
    df_results.to_csv('visualize/parameter_shift_results_fast.csv', index=False)
    print(f"\nSelesai! {len(df_results)} baris data berhasil diproses.")