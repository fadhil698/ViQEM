import os
import argparse
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from old.UnifiedPipeline import UnifiedPipeline
from typing import Optional
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from custom_noise import CustomNoiseBuilder

# =========================
# 1. DATABASE OBSERVABLES
# =========================

def get_observable_sets(type_mode: str):
    """
    type_mode: Gabungan mapper dan mode, contoh: 'jw_single', 'bk_dual'
    return: (mapper, grouping, list_of_observable_lists)
    """
    
    observables_db = {
        # JORDAN-WIGNER
        "jw_all": [['IIII', 'IIIZ', 'IIZI', 'IIZZ', 'IZII', 'IZIZ', 'ZIII', 'ZIIZ', 'YYYY', 'XXYY', 'YYXX', 'XXXX', 'IZZI', 'ZIZI', 'ZZII']],
        "jw_single": [['IIII'], ['IIIZ'], ['IIZI'], ['IIZZ'], ['IZII'], ['IZIZ'], ['ZIII'], ['ZIIZ'], ['YYYY'], ['XXYY'], ['YYXX'], ['XXXX'], ['IZZI'], ['ZIZI'], ['ZZII']],
        "jw_dual": [['IIII', 'IIIZ'], ['IIZI', 'IIZZ'], ['IZII', 'IZIZ'], ['ZIII', 'ZIIZ'], ['YYYY', 'XXYY'], ['YYXX', 'XXXX'], ['IZZI', 'ZIZI'], ['ZZII']],
        "jw_triple": [['IIII', 'IIIZ', 'IIZI'], ['IIZZ', 'IZII', 'IZIZ'], ['ZIII', 'ZIIZ', 'YYYY'], ['XXYY', 'YYXX', 'XXXX'], ['IZZI', 'ZIZI', 'ZZII']],

        # BRAVYI-KITAEV
        "bk_all": [['IIII', 'IIIZ', 'IIZZ', 'IIZI', 'IZII', 'IZIZ', 'ZZZI', 'ZZZZ', 'ZXIX', 'IXZX', 'ZXZX', 'IXIX', 'IZZZ', 'ZZIZ', 'ZIZI']],
        "bk_single": [['IIII'], ['IIIZ'], ['IIZZ'], ['IIZI'], ['IZII'], ['IZIZ'], ['ZZZI'], ['ZZZZ'], ['ZXIX'], ['IXZX'], ['ZXZX'], ['IXIX'], ['IZZZ'], ['ZZIZ'], ['ZIZI']],
        "bk_dual": [['IIII', 'IIIZ'], ['IIZZ', 'IIZI'], ['IZII', 'IZIZ'], ['ZZZI', 'ZZZZ'], ['ZXIX', 'IXZX'], ['ZXZX', 'IXIX'], ['IZZZ', 'ZZIZ'], ['ZIZI']],
        "bk_triple": [['IIII', 'IIIZ', 'IIZZ'], ['IIZI', 'IZII', 'IZIZ'], ['ZZZI', 'ZZZZ', 'ZXIX'], ['IXZX', 'ZXZX', 'IXIX'], ['IZZZ', 'ZZIZ', 'ZIZI']],

        # PARITY
        "parity_all": [['II', 'IZ', 'ZI', 'ZZ', 'XX']],
        "parity_single": [['II'], ['IZ'], ['ZI'], ['ZZ'], ['XX']],
        "parity_dual": [['II', 'IZ'], ['ZI', 'ZZ'], ['XX']],
        "parity_triple": [['II', 'IZ', 'ZI'], ['ZZ', 'XX']]
    }

    mode_key = type_mode.lower()

    if mode_key not in observables_db:
        raise ValueError(f"Mode '{type_mode}' tidak ditemukan di database.")

    # Parsing key untuk mendapatkan mapper dan grouping
    mapper, grouping = mode_key.split('_')
    
    return mapper, grouping, observables_db[mode_key]

# =========================
# 2. HELPER UNTUK PENAMAAN
# =========================

def get_experiment_codes(mapper: str, grouping: str):
    """
    Mengembalikan kode singkatan sesuai permintaan:
    Mapper: jw->J, bk->BK, parity->P
    Card: single->S, dual->D, triple->T, all->A
    """
    map_code = {
        "jw": "J",
        "bk": "BK",
        "parity": "P"
    }
    
    group_code = {
        "single": "S",
        "dual": "D",
        "triple": "T",
        "all": "A"
    }
    
    m_str = map_code.get(mapper, mapper.upper())
    g_str = group_code.get(grouping, grouping.upper())
    
    return m_str, g_str

# =========================
# 3. CORE RUNNER (Worker Function)
# =========================

def run_pipeline_for_mode(
    mode: str,
    n_qubits: int,
    n_circuits: int,
    bond_lengths: list,
    test_data_dir: str,
    base_output_dir: str,
    noise_model: Optional[NoiseModel] = None,  # <--- TAMBAHAN DI SINI
    data_condition: str = "custom"  # <--- ARGUMEN BARU
):
    """
    Fungsi ini akan dijalankan oleh setiap proses worker secara independen.
    """
    try:
        # 1. Ambil data sets
        mapper, grouping, obs_sets = get_observable_sets(mode)

        # --- LOGIKA DINAMIS PEMILIHAN FILE ---
        # Format: test_data_{mapper}_{kondisi}.csv
        # Contoh: test_data_jw_pitopi.csv
        filename = f"test_data_{mapper}_{data_condition}.csv"
            
        specific_test_data_path = os.path.join(test_data_dir, filename)

        # Cek apakah file ada (Best Practice untuk debugging)
        if not os.path.exists(specific_test_data_path):
            raise FileNotFoundError(f"File data tidak ditemukan: {specific_test_data_path}")

        # 2. Otomatisasi jumlah qubit
        actual_n_qubits = 2 if mapper == "parity" else n_qubits

        # 3. Setup Direktori Output
        specific_output_dir = os.path.join(f"{base_output_dir}_{mapper}", grouping)

        # 4. Ambil Kode Penamaan
        code_trans, code_card = get_experiment_codes(mapper, grouping)

        # Pesan awal (gunakan string formatted agar atomic saat print)
        print(f"[START] {mode.upper()} | Qubits: {actual_n_qubits} | Sets: {len(obs_sets)}")

        for idx, obs_list in enumerate(obs_sets, start=1):
            experiment_id = f"{code_trans}-{code_card}-{idx}"

            # Opsional: Comment out print ini jika terlalu berisik saat parallel
            # print(f"   -> Processing {mode}: {experiment_id}")

            pipeline = UnifiedPipeline(
                n_qubits=actual_n_qubits,
                n_circuits=n_circuits,
                experiment_id=experiment_id,
                output_dir=specific_output_dir, 
                noise_model = noise_model
            )

            pipeline.run_complete_workflow(
                observable_list=obs_list,
                test_data_path=specific_test_data_path,
                bond_lengths=bond_lengths,
                maxiter=125,
            )
        
        return f"SUCCESS: {mode}"

    except Exception as e:
        # Tangkap error agar tidak mematikan worker lain
        return f"FAILED: {mode} with error: {str(e)}"

# =========================
# 4. MAIN ARGUMENT PARSER
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run UnifiedPipeline parallel execution."
    )
    
    # Daftar mode individual yang sudah ada
    base_modes = [
        "jw_all", "jw_single", "jw_dual", "jw_triple",
        "bk_all", "bk_single", "bk_dual", "bk_triple",
        "parity_all", "parity_single", "parity_dual", "parity_triple"
    ]

    # Mode grup baru
    group_modes = ["jw", "bk", "parity", "all"]

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        # Gabungkan semua pilihan agar argparse tidak error
        choices=base_modes + group_modes, 
        help="Pilih mode individual (misal 'jw_single') atau grup ('jw', 'bk', 'parity', 'all').",
    )
    # ... (sisa argumen lain sama seperti sebelumnya) ...
    parser.add_argument("--n_qubits", type=int, default=4, help="Jumlah qubit dasar.")
    parser.add_argument("--n_circuits", type=int, default=500, help="Jumlah circuit.")
    parser.add_argument("--bond_lengths", type=float, nargs="+", default=[0.25, 0.45, 0.65, 0.725, 0.80, 1.50, 2.50, 3.00], help="List bond length.")
    parser.add_argument("--test_data_dir", type=str, default="input/test_data", help="Path folder csv.")
    parser.add_argument("--base_output_dir", type=str, default="unified_output", help="Output dir.")
    parser.add_argument("--max_workers", type=int, default=12, help="Max parallel process.")
    parser.add_argument(
        "--noise_qubit", 
        type=int, 
        nargs="+", 
        default=[0, 1, 2, 3], 
        help="List target noisy qubits (misal: --noise_qubit 0 1 2)")
    parser.add_argument(
            "--data_condition",
            type=str,
            default="custom",
            help="Kondisi data untuk nama file (misal: 'custom', 'pitopi', 'belem', '24'). File harus bernama test_data_{mapper}_{kondisi}.csv"
        )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Timer start
    start_time = time.time()

    # --- DEFINISI GRUP MODE ---
    # Kita definisikan batch job di sini untuk kemudahan
    mode_groups = {
        "jw":     ["jw_single", "jw_dual", "jw_triple", "jw_all"],
        "bk":     ["bk_single", "bk_dual", "bk_triple", "bk_all"],
        "parity": ["parity_single", "parity_dual", "parity_triple", "parity_all"]
    }

    # Gabungkan semua untuk mode 'all'
    all_modes_flat = mode_groups["jw"] + mode_groups["bk"] + mode_groups["parity"]

    # --- LOGIKA PEMILIHAN MODE ---
    target = args.mode.lower()
    modes_to_run = []

    if target == "all":
        modes_to_run = all_modes_flat
    elif target in mode_groups:
        # Jika user memilih 'jw', 'bk', atau 'parity'
        modes_to_run = mode_groups[target]
    else:
        # Jika user memilih mode spesifik, misal 'jw_single'
        modes_to_run = [target]

    print(f"{'='*60}")
    print(f"🚀 PARALLEL PIPELINE START")
    print(f"Target Mode: {args.mode.upper()}")
    print(f"Queue Size : {len(modes_to_run)} variations")
    print(f"Max Workers: {args.max_workers}")
    print(f"{'='*60}\n")

    # Definisikan Noise Custom
    # builder = CustomNoiseBuilder()

    # Noise Fake Backend
    backend_fake = FakeBelemV2()
    noise_model = NoiseModel.from_backend(backend_fake)
    
    # EKSEKUSI PARALEL (Sama seperti sebelumnya)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_mode = {
            executor.submit(
                run_pipeline_for_mode, 
                m, 
                args.n_qubits, 
                args.n_circuits, 
                args.bond_lengths, 
                args.test_data_dir, 
                args.base_output_dir,
                noise_model,
                # builder.get_noise_model(target_qubits=args.noise_qubit),
                args.data_condition
            ): m for m in modes_to_run
        }

        for future in as_completed(future_to_mode):
            mode_name = future_to_mode[future]
            try:
                result = future.result()
                print(f"✅ {result}")
            except Exception as exc:
                print(f"❌ ERROR pada {mode_name}: {exc}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 SELURUH BATCH ({args.mode}) SELESAI dalam {elapsed:.2f} detik.")
    print(f"{'='*60}")