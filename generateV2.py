import os
import argparse
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from ML_QEM.UnifiedPipeline import UnifiedPipeline

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
    base_output_dir: str
):
    """
    Fungsi ini akan dijalankan oleh setiap proses worker secara independen.
    """
    try:
        # 1. Ambil data sets
        mapper, grouping, obs_sets = get_observable_sets(mode)

        # Logika memilih file
        test_data_files = {
            "jw": "test_data_jw24.csv",
            "bk": "test_data_bk24.csv",
            "parity": "test_data_par24.csv"
        }
        
        filename = test_data_files.get(mapper)
        if not filename:
            raise ValueError(f"Mapper '{mapper}' tidak dikenali.")
            
        specific_test_data_path = os.path.join(test_data_dir, filename)

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
                output_dir=specific_output_dir 
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
    
    valid_modes = [
        "jw_all", "jw_single", "jw_dual", "jw_triple",
        "bk_all", "bk_single", "bk_dual", "bk_triple",
        "parity_all", "parity_single", "parity_dual", "parity_triple"
    ]

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=valid_modes + ["all"],
        help="Pilih 'all' untuk menjalankan SEMUA secara paralel.",
    )
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=4,
        help="Jumlah qubit dasar (default: 4).",
    )
    parser.add_argument(
        "--n_circuits",
        type=int,
        default=2000,
        help="Jumlah circuit training.",
    )
    parser.add_argument(
        "--bond_lengths",
        type=float,
        nargs="+",
        default=[0.25, 0.45, 0.65, 0.725, 0.80, 1.50, 2.50, 3.00],
        help="List bond length.",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="input/test_data",
        help="Path ke FOLDER yang berisi file csv.",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="unified_output",
        help="Nama dasar folder output.",
    )
    # ARGUMEN BARU UNTUK PARALLEL
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Jumlah proses paralel maksimal. Jangan melebihi jumlah core CPU.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Timer start
    start_time = time.time()

    # Tentukan list mode yang akan dijalankan
    modes_to_run = []
    if args.mode.lower() == "all":
        modes_to_run = [
            "jw_single", "jw_dual", "jw_triple", "jw_all",
            "bk_single", "bk_dual", "bk_triple", "bk_all",
            "parity_single", "parity_dual", "parity_triple", "parity_all"
        ]
    else:
        modes_to_run = [args.mode]

    print(f"{'='*60}")
    print(f"🚀 PARALLEL PIPELINE START")
    print(f"Modes: {len(modes_to_run)} items")
    print(f"Max Workers: {args.max_workers}")
    print(f"{'='*60}\n")

    # EKSEKUSI PARALEL
    # ProcessPoolExecutor sangat cocok untuk CPU-bound tasks (seperti komputasi matriks/QML)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit tasks ke executor
        future_to_mode = {
            executor.submit(
                run_pipeline_for_mode, 
                m, 
                args.n_qubits, 
                args.n_circuits, 
                args.bond_lengths, 
                args.test_data_dir, 
                args.base_output_dir
            ): m for m in modes_to_run
        }

        # Monitor progress saat task selesai (as_completed)
        for future in as_completed(future_to_mode):
            mode_name = future_to_mode[future]
            try:
                result = future.result()
                print(f"✅ {result}")
            except Exception as exc:
                print(f"❌ ERROR pada {mode_name}: {exc}")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🏁 SELURUH BATCH SELESAI dalam {elapsed:.2f} detik.")
    print(f"{'='*60}")