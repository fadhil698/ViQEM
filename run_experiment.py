"""
================================================================================
run_experiment.py — Unified ML-QEM Experiment Runner
================================================================================

Menggabungkan generate_test_data.py dan generate.py menjadi satu file utuh.

Fitur utama:
  - Dataset dibuat SATU KALI dan di-cache; hanya diregenerasi jika parameter
    eksperimen berubah (fingerprint-based caching).
  - Semua parameter penting dapat dikonfigurasi lewat ExperimentConfig.
  - Mendukung eksekusi paralel untuk dataset generation dan pipeline.

Parameter yang dapat dikonfigurasi:
  [Ansatz]    rotation_blocks, entanglement, reps
  [Noise]     noise_type ("fake_belem" | "fake_athens" | "custom"), target_qubits
  [ML/Data]   n_circuits, theta_range (min, max)
  [VQE]       bond_lengths, maxiter

Penggunaan:
  python run_experiment.py                          # jalankan semua mode
  python run_experiment.py --mode jw                # grup JW saja
  python run_experiment.py --mode jw_single         # mode spesifik
  python run_experiment.py --force-regen-data       # paksa regenerasi dataset
  python run_experiment.py --only-data              # generate data saja, tanpa pipeline
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeAthensV2, FakeBelemV2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_experiment")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

def load_config_file(path: str) -> Dict[str, str]:
    """Baca file config .txt format key = value."""
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            params[key.strip()] = value.strip()
    return params


def apply_config_file(args: argparse.Namespace, config_path: str) -> argparse.Namespace:
    """Override args dengan nilai dari config file."""
    raw = load_config_file(config_path)
    list_fields = {"rotation_blocks", "custom_noise_qubits", "bond_lengths"}
    int_fields = {"reps", "n_circuits", "maxiter", "max_workers_data", "max_workers_pipeline"}
    float_fields = {"theta_min", "theta_max"}
    int_list_fields = {"custom_noise_qubits"}
    float_list_fields = {"bond_lengths"}

    for key, val in raw.items():
        if key in list_fields:
            if key in int_list_fields:
                setattr(args, key.replace("-", "_"), [int(x) for x in val.split()])
            elif key in float_list_fields:
                setattr(args, key.replace("-", "_"), [float(x) for x in val.split()])
            else:
                setattr(args, key.replace("-", "_"), val.split())
        elif key in int_fields:
            setattr(args, key.replace("-", "_"), int(val))
        elif key in float_fields:
            setattr(args, key.replace("-", "_"), float(val))
        else:
            setattr(args, key.replace("-", "_"), val)
    return args

def log_run_to_file(cfg: ExperimentConfig, args: argparse.Namespace, 
                    elapsed: float, status: str = "COMPLETED"):
    """Catat detail setiap run ke file log permanen."""
    log_path = "run_history.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"TIMESTAMP    : {timestamp}\n")
        f.write(f"STATUS       : {status}\n")
        f.write(f"MODE         : {args.mode}\n")
        f.write(f"KONDISI      : {cfg.get_condition_label()}\n")
        f.write(f"FINGERPRINT  : {cfg.fingerprint()}\n")
        f.write(f"OUTPUT DIR   : {cfg.base_output_dir}\n")
        f.write(f"DATA DIR     : {cfg.test_data_dir}\n")
        f.write(f"--- Ansatz ---\n")
        f.write(f"rotation_blocks     : {cfg.rotation_blocks}\n")
        f.write(f"entanglement        : {cfg.entanglement}\n")
        f.write(f"entanglement_blocks : {cfg.entanglement_blocks}\n")
        f.write(f"reps                : {cfg.reps}\n")
        f.write(f"--- Noise ---\n")
        f.write(f"noise_type          : {cfg.noise_type}\n")
        f.write(f"custom_noise_qubits : {cfg.custom_noise_qubits}\n")
        f.write(f"--- Dataset ---\n")
        f.write(f"n_circuits          : {cfg.n_circuits}\n")
        f.write(f"theta_range         : [{cfg.theta_min}, {cfg.theta_max}]\n")
        f.write(f"--- VQE ---\n")
        f.write(f"bond_lengths        : {cfg.bond_lengths}\n")
        f.write(f"maxiter             : {cfg.maxiter}\n")
        f.write(f"--- Waktu ---\n")
        f.write(f"elapsed             : {elapsed:.2f} detik\n")

@dataclass
class ExperimentConfig:
    """
    Semua parameter eksperimen terpusat di sini.
    Mengubah nilai apapun akan otomatis memicu regenerasi dataset.
    """

    # ── Ansatz ──────────────────────────────────────────────────────────────
    rotation_blocks: List[str] = field(default_factory=lambda: ["ry"])
    # Gate rotasi ansatz. Contoh: ["ry"], ["rx", "ry"], ["rz", "ry"]

    entanglement: str = "linear"
    # Pola entanglement: "linear" | "full" | "circular" | "sca"

    entanglement_blocks: str = "cx"
    # Gate entanglement: "cx" | "cz" | "ecr"

    reps: int = 2
    # Jumlah layer repetisi ansatz

    insert_barriers: bool = True

    # ── Noise ───────────────────────────────────────────────────────────────
    noise_type: str = "fake_belem"
    # Jenis noise model:
    #   "fake_belem"  → NoiseModel dari FakeBelemV2
    #   "fake_athens" → NoiseModel dari FakeAthensV2
    #   "custom"      → CustomNoiseBuilder (lihat kode custom_noise.py)

    custom_noise_qubits: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # Qubit target untuk custom noise (hanya digunakan jika noise_type="custom")

    # ── Dataset / ML ─────────────────────────────────────────────────────────
    n_circuits: int = 500
    # Jumlah data point (baris) yang dibuat per mapper

    theta_min: float = -np.pi
    theta_max: float = np.pi
    # Range sampling parameter theta (uniform random)

    # ── Pipeline / VQE ───────────────────────────────────────────────────────
    bond_lengths: List[float] = field(
        default_factory=lambda: [0.25, 0.45, 0.65, 0.725, 0.80, 1.50, 2.50, 3.00]
    )
    maxiter: int = 125
    # Iterasi maksimum SPSA

    # ── I/O ──────────────────────────────────────────────────────────────────
    test_data_dir: str = "input/test_data"
    base_output_dir: str = "unified_output"

    # ── Paralelisme ──────────────────────────────────────────────────────────
    max_workers_data: int = 8
    # Jumlah worker untuk generate dataset

    max_workers_pipeline: int = 12
    # Jumlah worker untuk pipeline (training + VQE)

    # ── Kondisi label (untuk nama file) ──────────────────────────────────────
    data_condition: str = "custom"
    # Label kondisi eksperimen. Akan digunakan dalam nama file CSV.
    # Jika None, nama file diturunkan otomatis dari fingerprint config.

    def fingerprint(self) -> str:
        """
        SHA-256 hash dari seluruh parameter yang mempengaruhi isi dataset.
        Digunakan untuk memutuskan apakah dataset perlu diregenerasi.
        """
        relevant = {
            "rotation_blocks": self.rotation_blocks,
            "entanglement": self.entanglement,
            "entanglement_blocks": self.entanglement_blocks,
            "reps": self.reps,
            "noise_type": self.noise_type,
            "custom_noise_qubits": self.custom_noise_qubits,
            "n_circuits": self.n_circuits,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
        }
        raw = json.dumps(relevant, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()[:12]

    def get_condition_label(self) -> str:
        """Label kondisi yang digunakan dalam nama file CSV."""
        return self.data_condition if self.data_condition else f"fp_{self.fingerprint()}"

    def get_csv_path(self, mapper: str) -> str:
        label = self.get_condition_label()
        return os.path.join(self.test_data_dir, f"test_data_{mapper}_{label}.csv")

    def get_meta_path(self, mapper: str) -> str:
        label = self.get_condition_label()
        return os.path.join(self.test_data_dir, f".meta_{mapper}_{label}.json")


# ============================================================================
# OBSERVABLE SETS DATABASE
# ============================================================================

def get_observable_sets(type_mode: str) -> Tuple[str, str, List[List[str]]]:
    """
    type_mode: gabungan mapper dan grouping, contoh: 'jw_single', 'bk_dual'
    Returns: (mapper, grouping, list_of_observable_lists)
    """
    observables_db = {
        # JORDAN-WIGNER
        "jw_all":    [['IIII', 'IIIZ', 'IIZI', 'IIZZ', 'IZII', 'IZIZ', 'ZIII', 'ZIIZ',
                        'YYYY', 'XXYY', 'YYXX', 'XXXX', 'IZZI', 'ZIZI', 'ZZII']],
        "jw_single": [['IIII'], ['IIIZ'], ['IIZI'], ['IIZZ'], ['IZII'], ['IZIZ'], ['ZIII'],
                      ['ZIIZ'], ['YYYY'], ['XXYY'], ['YYXX'], ['XXXX'], ['IZZI'], ['ZIZI'], ['ZZII']],
        "jw_dual":   [['IIII', 'IIIZ'], ['IIZI', 'IIZZ'], ['IZII', 'IZIZ'], ['ZIII', 'ZIIZ'],
                      ['YYYY', 'XXYY'], ['YYXX', 'XXXX'], ['IZZI', 'ZIZI'], ['ZZII']],
        "jw_triple": [['IIII', 'IIIZ', 'IIZI'], ['IIZZ', 'IZII', 'IZIZ'], ['ZIII', 'ZIIZ', 'YYYY'],
                      ['XXYY', 'YYXX', 'XXXX'], ['IZZI', 'ZIZI', 'ZZII']],
        # BRAVYI-KITAEV
        "bk_all":    [['IIII', 'IIIZ', 'IIZZ', 'IIZI', 'IZII', 'IZIZ', 'ZZZI', 'ZZZZ',
                        'ZXIX', 'IXZX', 'ZXZX', 'IXIX', 'IZZZ', 'ZZIZ', 'ZIZI']],
        "bk_single": [['IIII'], ['IIIZ'], ['IIZZ'], ['IIZI'], ['IZII'], ['IZIZ'], ['ZZZI'],
                      ['ZZZZ'], ['ZXIX'], ['IXZX'], ['ZXZX'], ['IXIX'], ['IZZZ'], ['ZZIZ'], ['ZIZI']],
        "bk_dual":   [['IIII', 'IIIZ'], ['IIZZ', 'IIZI'], ['IZII', 'IZIZ'], ['ZZZI', 'ZZZZ'],
                      ['ZXIX', 'IXZX'], ['ZXZX', 'IXIX'], ['IZZZ', 'ZZIZ'], ['ZIZI']],
        "bk_triple": [['IIII', 'IIIZ', 'IIZZ'], ['IIZI', 'IZII', 'IZIZ'], ['ZZZI', 'ZZZZ', 'ZXIX'],
                      ['IXZX', 'ZXZX', 'IXIX'], ['IZZZ', 'ZZIZ', 'ZIZI']],
        # PARITY
        "parity_all":    [['II', 'IZ', 'ZI', 'ZZ', 'XX']],
        "parity_single": [['II'], ['IZ'], ['ZI'], ['ZZ'], ['XX']],
        "parity_dual":   [['II', 'IZ'], ['ZI', 'ZZ'], ['XX']],
        "parity_triple": [['II', 'IZ', 'ZI'], ['ZZ', 'XX']],
    }

    key = type_mode.lower()
    if key not in observables_db:
        raise ValueError(f"Mode '{type_mode}' tidak ditemukan. Pilihan: {list(observables_db.keys())}")

    mapper, grouping = key.split("_", 1)
    return mapper, grouping, observables_db[key]


# Semua observable per mapper untuk generate dataset
ALL_OBSERVABLES = {
    "jw":     ['IIII', 'IIIZ', 'IIZI', 'IIZZ', 'IZII', 'IZIZ', 'ZIII', 'ZIIZ',
               'YYYY', 'XXYY', 'YYXX', 'XXXX', 'IZZI', 'ZIZI', 'ZZII'],
    "bk":     ['IIII', 'IIIZ', 'IIZZ', 'IIZI', 'IZII', 'IZIZ', 'ZZZI', 'ZZZZ',
               'ZXIX', 'IXZX', 'ZXZX', 'IXIX', 'IZZZ', 'ZZIZ', 'ZIZI'],
    "parity": ['II', 'IZ', 'ZI', 'ZZ', 'XX'],
}


def get_experiment_codes(mapper: str, grouping: str) -> Tuple[str, str]:
    map_code   = {"jw": "J",  "bk": "BK", "parity": "P"}
    group_code = {"single": "S", "dual": "D", "triple": "T", "all": "A"}
    return map_code.get(mapper, mapper.upper()), group_code.get(grouping, grouping.upper())


# ============================================================================
# NOISE MODEL FACTORY
# ============================================================================

def build_noise_model(cfg: ExperimentConfig) -> NoiseModel:
    """Bangun noise model sesuai konfigurasi."""
    if cfg.noise_type == "fake_belem":
        return NoiseModel.from_backend(FakeBelemV2())
    elif cfg.noise_type == "fake_athens":
        return NoiseModel.from_backend(FakeAthensV2())
    elif cfg.noise_type == "custom":
        from custom_noise import CustomNoiseBuilder
        return CustomNoiseBuilder().get_noise_model(target_qubits=cfg.custom_noise_qubits)
    else:
        raise ValueError(
            f"noise_type tidak dikenal: '{cfg.noise_type}'. "
            "Pilihan: 'fake_belem', 'fake_athens', 'custom'."
        )


# ============================================================================
# DATASET GENERATION
# ============================================================================

def _dataset_is_valid(cfg: ExperimentConfig, mapper: str) -> bool:
    """
    Cek apakah dataset yang sudah ada masih valid (fingerprint cocok dan file ada).
    """
    csv_path  = cfg.get_csv_path(mapper)
    meta_path = cfg.get_meta_path(mapper)

    if not os.path.exists(csv_path) or not os.path.exists(meta_path):
        return False

    with open(meta_path) as f:
        saved_meta = json.load(f)

    return saved_meta.get("fingerprint") == cfg.fingerprint()


def _save_dataset_meta(cfg: ExperimentConfig, mapper: str, n_rows: int):
    """Simpan metadata dataset untuk keperluan cache."""
    meta = {
        "fingerprint": cfg.fingerprint(),
        "mapper": mapper,
        "n_circuits": cfg.n_circuits,
        "n_rows_actual": n_rows,
        "noise_type": cfg.noise_type,
        "rotation_blocks": cfg.rotation_blocks,
        "entanglement": cfg.entanglement,
        "entanglement_blocks": cfg.entanglement_blocks,
        "reps": cfg.reps,
        "theta_range": [cfg.theta_min, cfg.theta_max],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cfg.get_meta_path(mapper), "w") as f:
        json.dump(meta, f, indent=2)


def _generate_single_row(
    idx: int,
    mapper: str,
    obs_list: List[str],
    n_qubits: int,
    n_params: int,
    noise_model: Any,
    theta_min: float,
    theta_max: float,
    rotation_blocks: List[str],
    entanglement: str,
    entanglement_blocks: str,
    reps: int,
) -> Dict[str, Any]:
    """
    Worker function: hitung satu baris data (ideal, noisy, ZNE).
    Dijalankan secara paralel di dalam ProcessPoolExecutor.
    """
    # Import lokal agar bisa berjalan di dalam worker process
    from qiskit.circuit.library import n_local
    from qiskit.quantum_info import SparsePauliOp as _SPO
    from ML_QEM.qem_pipeline import AnsatzConfig, CalculateEnergy

    n_obs = len(obs_list)

    # Buat ansatz sesuai konfigurasi
    ansatz = n_local(
        n_qubits,
        rotation_blocks=rotation_blocks,
        entanglement_blocks=entanglement_blocks,
        entanglement=entanglement,
        reps=reps,
        insert_barriers=True,
    )

    energy_calc = CalculateEnergy(n_qubits=n_qubits, noise_model=noise_model)

    pauli   = obs_list[idx % n_obs]
    obs_op  = _SPO(pauli)
    theta   = np.random.uniform(theta_min, theta_max, size=n_params)

    ideal_e = energy_calc.calculate_ideal_energy(ansatz, obs_op, theta)
    noisy_e = energy_calc.calculate_noisy_energy(ansatz, obs_op, theta)
    zne_e   = energy_calc.calculate_zne_energy(ansatz, obs_op, theta)

    row: Dict[str, Any] = {
        "noisy_energy": float(noisy_e),
        "ideal_energy": float(ideal_e),
        "zne_energy":   float(zne_e),
        "observable":   pauli,
    }
    for j, p in enumerate(theta):
        row[f"param_{j}"] = float(p)

    return row


def generate_dataset_for_mapper(
    mapper: str,
    cfg: ExperimentConfig,
    noise_model: NoiseModel,
    force: bool = False,
) -> str:
    """
    Generate dataset untuk satu mapper.
    Jika sudah ada dan valid, lewati (skip).
    Returns: path ke file CSV.
    """
    csv_path = cfg.get_csv_path(mapper)

    if not force and _dataset_is_valid(cfg, mapper):
        logger.info(f"[CACHE HIT] Dataset '{mapper}' masih valid → skip regenerasi.")
        logger.info(f"  Path: {csv_path}")
        return csv_path

    logger.info(f"[GENERATE] Dataset '{mapper}' — {cfg.n_circuits} baris ...")
    os.makedirs(cfg.test_data_dir, exist_ok=True)

    obs_list  = ALL_OBSERVABLES[mapper]
    n_qubits  = 2 if mapper == "parity" else 4

    # Hitung jumlah parameter ansatz
    from qiskit.circuit.library import n_local
    template = n_local(
        n_qubits,
        rotation_blocks=cfg.rotation_blocks,
        entanglement_blocks=cfg.entanglement_blocks,
        entanglement=cfg.entanglement,
        reps=cfg.reps,
    )
    n_params = template.num_parameters
    logger.info(f"  Ansatz: {n_qubits} qubits, {n_params} params, reps={cfg.reps}, "
                f"rot={cfg.rotation_blocks}, ent={cfg.entanglement}/{cfg.entanglement_blocks}")

    results = []
    with ProcessPoolExecutor(max_workers=cfg.max_workers_data) as executor:
        futures = [
            executor.submit(
                _generate_single_row,
                i, mapper, obs_list, n_qubits, n_params, noise_model,
                cfg.theta_min, cfg.theta_max,
                cfg.rotation_blocks, cfg.entanglement,
                cfg.entanglement_blocks, cfg.reps,
            )
            for i in range(cfg.n_circuits)
        ]

        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 200 == 0 or done == cfg.n_circuits:
                logger.info(f"  {mapper.upper()}: {done}/{cfg.n_circuits} selesai")

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    _save_dataset_meta(cfg, mapper, len(df))
    logger.info(f"✅ Dataset '{mapper}' disimpan → {csv_path}")
    return csv_path


def generate_all_datasets(cfg: ExperimentConfig, force: bool = False):
    """
    Generate dataset untuk semua mapper (jw, bk, parity).
    Dataset yang masih valid tidak akan diregenerasi.
    """
    logger.info("=" * 60)
    logger.info("🗄  DATASET GENERATION")
    logger.info(f"   Kondisi  : {cfg.get_condition_label()}")
    logger.info(f"   Fingerprint: {cfg.fingerprint()}")
    logger.info("=" * 60)

    noise_model = build_noise_model(cfg)
    paths = {}
    for mapper in ["jw", "bk", "parity"]:
        paths[mapper] = generate_dataset_for_mapper(mapper, cfg, noise_model, force=force)

    logger.info("\n✅ Semua dataset siap.")
    return paths


# ============================================================================
# PIPELINE RUNNER
# ============================================================================

def run_pipeline_for_mode(
    mode: str,
    cfg_dict: Dict,  # ExperimentConfig sebagai dict agar bisa di-pickle
    noise_model: NoiseModel,
) -> str:
    """
    Worker function: jalankan UnifiedPipeline untuk satu mode.
    Menerima cfg_dict (bukan objek dataclass) agar aman di-pickle lintas proses.
    """
    try:
        cfg = ExperimentConfig(**cfg_dict)
        mapper, grouping, obs_sets = get_observable_sets(mode)

        csv_path = cfg.get_csv_path(mapper)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset tidak ditemukan: {csv_path}")

        actual_n_qubits  = 2 if mapper == "parity" else 4
        specific_out_dir = os.path.join(f"{cfg.base_output_dir}_{mapper}", grouping)
        code_trans, code_card = get_experiment_codes(mapper, grouping)

        # Import di sini agar worker bersih
        from ML_QEM.qem_pipeline import UnifiedPipeline, AnsatzConfig

        # Bangun AnsatzConfig dari ExperimentConfig — inilah yang diteruskan ke pipeline
        ansatz_cfg = AnsatzConfig(
            n_qubits=actual_n_qubits,
            rotation_blocks=cfg.rotation_blocks,
            entanglement_blocks=cfg.entanglement_blocks,
            entanglement=cfg.entanglement,
            reps=cfg.reps,
        )

        logger.info(f"[START] {mode.upper()} | Qubits: {actual_n_qubits} | Sets: {len(obs_sets)}")

        for idx, obs_list in enumerate(obs_sets, start=1):
            experiment_id = f"{code_trans}-{code_card}-{idx}"

            pipeline = UnifiedPipeline(
                n_qubits=actual_n_qubits,
                n_circuits=cfg.n_circuits,
                experiment_id=experiment_id,
                output_dir=specific_out_dir,
                noise_model=noise_model,
                ansatz_config=ansatz_cfg,
                theta_range=(cfg.theta_min, cfg.theta_max),
            )

            pipeline.run_complete_workflow(
                observable_list=obs_list,
                test_data_path=csv_path,
                bond_lengths=cfg.bond_lengths,
                maxiter=cfg.maxiter,
            )

        return f"SUCCESS: {mode}"

    except Exception as e:
        import traceback
        return f"FAILED: {mode} | {str(e)}\n{traceback.format_exc()}"


# ============================================================================
# MODE RESOLVER
# ============================================================================

MODE_GROUPS = {
    "jw":     ["jw_single", "jw_dual", "jw_triple", "jw_all"],
    "bk":     ["bk_single", "bk_dual", "bk_triple", "bk_all"],
    "parity": ["parity_single", "parity_dual", "parity_triple", "parity_all"],
}
ALL_MODES_FLAT = MODE_GROUPS["jw"] + MODE_GROUPS["bk"] + MODE_GROUPS["parity"]


def resolve_modes(mode_arg: str) -> List[str]:
    t = mode_arg.lower()
    if t == "all":
        return ALL_MODES_FLAT
    elif t in MODE_GROUPS:
        return MODE_GROUPS[t]
    else:
        if t not in ALL_MODES_FLAT:
            raise ValueError(f"Mode tidak dikenal: '{mode_arg}'")
        return [t]


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified ML-QEM Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python run_experiment.py                              # semua mode, auto-cache dataset
  python run_experiment.py --mode jw                   # grup JW saja
  python run_experiment.py --mode bk_dual              # mode spesifik
  python run_experiment.py --force-regen-data          # paksa regenerasi dataset
  python run_experiment.py --only-data                 # generate data saja
  python run_experiment.py --noise-type fake_athens    # ganti noise model
  python run_experiment.py --rotation-blocks ry rx     # ganti gate rotasi ansatz
  python run_experiment.py --entanglement full         # ganti pola entanglement
  python run_experiment.py --reps 3                    # tambah layer ansatz
  python run_experiment.py --n-circuits 2000           # lebih banyak data
  python run_experiment.py --theta-min -1.57 --theta-max 1.57   # range theta
        """
    )

    # Mode eksperimen
    p.add_argument("--mode", type=str, default="all",
                   help="Mode: 'all' | 'jw' | 'bk' | 'parity' | mode spesifik (contoh: 'jw_single')")

    # ── Ansatz ─────────────────────────────────────────────────────────────
    p.add_argument("--rotation-blocks", type=str, nargs="+", default=["ry"],
                   help="Gate rotasi ansatz. Contoh: --rotation-blocks ry  atau  --rotation-blocks rx ry")
    p.add_argument("--entanglement", type=str, default="linear",
                   choices=["linear", "full", "circular", "sca"],
                   help="Pola entanglement ansatz.")
    p.add_argument("--entanglement-blocks", type=str, default="cx",
                   choices=["cx", "cz", "ecr"],
                   help="Gate entanglement ansatz.")
    p.add_argument("--reps", type=int, default=2,
                   help="Jumlah layer (reps) ansatz.")

    # ── Noise ───────────────────────────────────────────────────────────────
    p.add_argument("--noise-type", type=str, default="fake_belem",
                   choices=["fake_belem", "fake_athens", "custom"],
                   help="Jenis noise model yang digunakan.")
    p.add_argument("--custom-noise-qubits", type=int, nargs="+", default=[0, 1, 2, 3],
                   help="Target qubit untuk custom noise (hanya jika --noise-type custom).")

    # ── Dataset / ML ─────────────────────────────────────────────────────────
    p.add_argument("--n-circuits", type=int, default=500,
                   help="Jumlah data point per mapper.")
    p.add_argument("--theta-min", type=float, default=-np.pi,
                   help="Batas bawah range parameter theta.")
    p.add_argument("--theta-max", type=float, default=np.pi,
                   help="Batas atas range parameter theta.")
    p.add_argument("--data-condition", type=str, default="custom",
                   help="Label kondisi eksperimen (digunakan dalam nama file CSV).")

    # ── VQE ──────────────────────────────────────────────────────────────────
    p.add_argument("--bond-lengths", type=float, nargs="+",
                   default=[0.25, 0.45, 0.65, 0.725, 0.80, 1.50, 2.50, 3.00],
                   help="Daftar bond length untuk VQE.")
    p.add_argument("--maxiter", type=int, default=125,
                   help="Iterasi maksimum SPSA untuk VQE.")

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("--test-data-dir", type=str, default="input/test_data",
                   help="Direktori untuk menyimpan/membaca file CSV dataset.")
    p.add_argument("--base-output-dir", type=str, default="unified_output",
                   help="Direktori output hasil pipeline.")

    # ── Paralelisme ──────────────────────────────────────────────────────────
    p.add_argument("--max-workers-data", type=int, default=8,
                   help="Jumlah worker paralel untuk generate dataset.")
    p.add_argument("--max-workers-pipeline", type=int, default=12,
                   help="Jumlah worker paralel untuk pipeline.")

    # ── Flow control ─────────────────────────────────────────────────────────
    p.add_argument("--force-regen-data", action="store_true",
                   help="Paksa regenerasi dataset meskipun cache valid.")
    p.add_argument("--only-data", action="store_true",
                   help="Hanya generate dataset, tidak menjalankan pipeline.")
    p.add_argument("--config", type=str, default=None,
               help="Path ke file config .txt (override semua argumen lain).")

    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    start_time = time.time()
    
    if args.config:
        args = apply_config_file(args, args.config)

    # ── Bangun ExperimentConfig dari argumen ──────────────────────────────
    cfg = ExperimentConfig(
        rotation_blocks      = args.rotation_blocks,
        entanglement         = args.entanglement,
        entanglement_blocks  = args.entanglement_blocks,
        reps                 = args.reps,
        noise_type           = args.noise_type,
        custom_noise_qubits  = args.custom_noise_qubits,
        n_circuits           = args.n_circuits,
        theta_min            = args.theta_min,
        theta_max            = args.theta_max,
        bond_lengths         = args.bond_lengths,
        maxiter              = args.maxiter,
        test_data_dir        = args.test_data_dir,
        base_output_dir      = args.base_output_dir,
        max_workers_data     = args.max_workers_data,
        max_workers_pipeline = args.max_workers_pipeline,
        data_condition       = args.data_condition,
    )
    
    try:
        # ... seluruh logika main() yang sudah ada ...

        modes_to_run = resolve_modes(args.mode)

        # ── Banner ───────────────────────────────────────────────────────────────
        print(f"\n{'='*65}")
        print(f"  🚀 UNIFIED ML-QEM EXPERIMENT RUNNER")
        print(f"{'='*65}")
        print(f"  Mode         : {args.mode.upper()} ({len(modes_to_run)} variasi)")
        print(f"  Kondisi      : {cfg.get_condition_label()}")
        print(f"  Fingerprint  : {cfg.fingerprint()}")
        print(f"  Ansatz       : rot={cfg.rotation_blocks}, ent={cfg.entanglement}/")
        print(f"                 {cfg.entanglement_blocks}, reps={cfg.reps}")
        print(f"  Noise        : {cfg.noise_type}")
        print(f"  n_circuits   : {cfg.n_circuits}")
        print(f"  theta range  : [{cfg.theta_min:.4f}, {cfg.theta_max:.4f}]")
        print(f"  bond_lengths : {cfg.bond_lengths}")
        print(f"  max_workers  : data={cfg.max_workers_data}, pipeline={cfg.max_workers_pipeline}")
        print(f"{'='*65}\n")

        # ── STEP 1: Generate / Validasi Dataset ──────────────────────────────────
        generate_all_datasets(cfg, force=args.force_regen_data)

        if args.only_data:
            elapsed = time.time() - start_time
            print(f"\n✅ Dataset generation selesai dalam {elapsed:.2f} detik.")
            return

        # ── STEP 2: Jalankan Pipeline ─────────────────────────────────────────────
        print(f"\n{'='*65}")
        print(f"  ⚙  PIPELINE EXECUTION")
        print(f"  Queue: {len(modes_to_run)} mode | Workers: {cfg.max_workers_pipeline}")
        print(f"{'='*65}\n")

        noise_model = build_noise_model(cfg)
        cfg_dict = asdict(cfg)  # dict agar bisa di-pickle lintas proses

        with ProcessPoolExecutor(max_workers=cfg.max_workers_pipeline) as executor:
            future_to_mode = {
                executor.submit(run_pipeline_for_mode, m, cfg_dict, noise_model): m
                for m in modes_to_run
            }

            for future in as_completed(future_to_mode):
                mode_name = future_to_mode[future]
                try:
                    result = future.result()
                    icon = "✅" if result.startswith("SUCCESS") else "❌"
                    print(f"  {icon} {result}")
                except Exception as exc:
                    print(f"  ❌ ERROR pada {mode_name}: {exc}")

        elapsed = time.time() - start_time
        print(f"\n{'='*65}")
        print(f"  🏁 SELESAI — mode '{args.mode}' dalam {elapsed:.2f} detik")
        print(f"{'='*65}\n")
        
        log_run_to_file(cfg, args, elapsed, status="COMPLETED")

        log_run_to_file(cfg, args, time.time() - start_time, "COMPLETED")
    except Exception as e:
        log_run_to_file(cfg, args, time.time() - start_time, f"FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
