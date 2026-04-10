"""
server.py - Covalent Workflow Orchestration untuk ML-QEM Pipeline

Fitur:
1. Workflow A: Pipeline dengan observable variations (paralel)
2. Workflow B: VQE untuk range bond lengths (paralel)
3. Dependency management dan status checking
4. Support untuk 2 executor: local dan remote (SLURM)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime

try:
    import covalent as ct
    COVALENT_AVAILABLE = True
except ImportError:
    COVALENT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Covalent not available. Install with: pip install covalent")

from qiskit.quantum_info import SparsePauliOp
from old.qem_main_updated_v2 import (
    GenerateAnsatz, AnsatzConfig, CalculateEnergy,
    TrainML, VQERunner, H2HamiltonianGenerator, ExtendedPipeline
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# EXECUTOR CONFIGURATION
# ============================================================================

def get_executor(executor_type: str = "local", **kwargs):
    """
    Get executor configuration.
    
    Parameters
    ----------
    executor_type : str
        "local" atau "remote"
    **kwargs
        Additional executor parameters
    
    Returns
    -------
    executor
        Configured Covalent executor
    """
    if not COVALENT_AVAILABLE:
        raise RuntimeError("Covalent not installed")
    
    if executor_type == "local":
        return ct.executor.LocalExecutor()
    
    elif executor_type == "remote":
        # SLURM remote executor
        slurm_config = {
            "credentials_file": kwargs.get("credentials_file", "~/.covalent/covalent_slurm_config"),
            "remote_workdir": kwargs.get("remote_workdir", "/tmp/covalent"),
            "job_name": kwargs.get("job_name", "qem_job"),
            "partition": kwargs.get("partition", "gpu"),
            "num_nodes": kwargs.get("num_nodes", 1),
            "cpus_per_task": kwargs.get("cpus_per_task", 8),
            "gpus_per_node": kwargs.get("gpus_per_node", 1),
            "time_limit": kwargs.get("time_limit", "01:00:00"),
            "username": kwargs.get("username", None),
            "address": kwargs.get("address", "localhost"),
            "ssh_key_file": kwargs.get("ssh_key_file", None),
        }
        
        return ct.executor.SlurmExecutor(**slurm_config)
    
    else:
        raise ValueError(f"Unknown executor type: {executor_type}")


# ============================================================================
# WORKFLOW A: PIPELINE DENGAN OBSERVABLE VARIATIONS (PARALEL)
# ============================================================================

@ct.electron(executor=ct.executor.LocalExecutor())
def prepare_dataset_task(
    n_circuits: int,
    n_qubits: int,
    observable: str
) -> Dict:
    """
    Electron: Generate dataset untuk satu observable.
    
    Parameters
    ----------
    n_circuits : int
        Number of circuits untuk observable ini
    n_qubits : int
        Number of qubits
    observable : str
        Pauli observable string
    
    Returns
    -------
    Dict
        Dataset hasil untuk observable ini
    """
    logger.info(f"Generating dataset for observable: {observable}")
    
    from old.qem_main_updated_v2 import GenerateAnsatz, AnsatzConfig, CalculateEnergy
    
    # Create components
    config = AnsatzConfig(n_qubits=n_qubits)
    ansatz_gen = GenerateAnsatz(config)
    base_ansatz = ansatz_gen.create_ansatz()
    n_params = base_ansatz.num_parameters
    
    energy_calc = CalculateEnergy(n_qubits=n_qubits)
    
    # Generate data
    noisy_energies = []
    ideal_energies = []
    zne_energies = []
    parameters = []
    
    for idx in range(n_circuits):
        theta = np.random.uniform(-5.0, 5.0, size=n_params)
        obs_op = SparsePauliOp(observable)
        
        ideal_e = energy_calc.calculate_ideal_energy(base_ansatz, obs_op, theta)
        noisy_e = energy_calc.calculate_noisy_energy(base_ansatz, obs_op, theta)
        zne_e = energy_calc.calculate_zne_energy(base_ansatz, obs_op, theta)
        
        ideal_energies.append(float(ideal_e))
        noisy_energies.append(float(noisy_e))
        zne_energies.append(float(zne_e))
        parameters.append(theta)
        
        if (idx + 1) % (n_circuits // 5 + 1) == 0:
            logger.info(f"{idx+1}/{n_circuits} circuits for {observable}")
    
    return {
        "observable": observable,
        "noisy_energies": np.array(noisy_energies),
        "ideal_energies": np.array(ideal_energies),
        "zne_energies": np.array(zne_energies),
        "parameters": np.array(parameters)
    }


@ct.electron(executor=ct.executor.LocalExecutor())
def train_model_task(
    observable_results: List[Dict],
    n_qubits: int,
    model_id: int
) -> Dict:
    """
    Electron: Train ML model dari dataset yang telah di-collect.
    
    Parameters
    ----------
    observable_results : List[Dict]
        Hasil dari semua observable tasks
    n_qubits : int
        Number of qubits
    model_id : int
        Model identifier
    
    Returns
    -------
    Dict
        Training metrics
    """
    logger.info(f"Training model from {len(observable_results)} observables")
    
    from old.qem_main_updated_v2 import TrainML
    
    # Aggregate data dari semua observables
    all_noisy = []
    all_ideal = []
    all_zne = []
    all_obs = []
    all_params = []
    
    for result in observable_results:
        all_noisy.extend(result["noisy_energies"])
        all_ideal.extend(result["ideal_energies"])
        all_zne.extend(result["zne_energies"])
        all_params.extend(result["parameters"])
        
        obs = result["observable"]
        all_obs.extend([obs] * len(result["noisy_energies"]))
    
    # Train model
    ml_trainer = TrainML()
    X, y = ml_trainer.build_dataset(
        np.array(all_noisy),
        np.array(all_ideal),
        all_obs,
        np.array(all_params)
    )
    
    metrics = ml_trainer.train_model(X, y, n_estimators=100)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/qem_model_{model_id}.pkl"
    ml_trainer.save_model(model_path)
    
    metrics["model_path"] = model_path
    logger.info(f"✓ Model trained and saved: {model_path}")
    
    return metrics


@ct.electron(executor=ct.executor.LocalExecutor())
def check_workflow_a_status(
    training_results: Dict,
    threshold_r2: float = 0.8
) -> Dict:
    """
    Electron: Check status workflow A.
    
    Parameters
    ----------
    training_results : Dict
        Hasil training dari workflow A
    threshold_r2 : float
        Threshold R² untuk pass/fail
    
    Returns
    -------
    Dict
        Status dan decision untuk workflow B
    """
    logger.info("Checking Workflow A status...")
    
    test_r2 = training_results.get("test_r2", 0.0)
    passed = test_r2 >= threshold_r2
    
    status = {
        "passed": passed,
        "test_r2": test_r2,
        "threshold": threshold_r2,
        "message": f"Workflow A: {'PASSED ✓' if passed else 'FAILED ✗'} (R²={test_r2:.4f})",
        "model_path": training_results.get("model_path"),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(status["message"])
    return status


@ct.lattice
def workflow_a_pipeline(
    observable_list: List[str],
    n_circuits_per_obs: int = 500,
    n_qubits: int = 4,
    model_id: int = 1,
    threshold_r2: float = 0.8
) -> Dict:
    """
    Workflow A: Pipeline dengan observable variations (paralel).
    
    Fitur:
    - Generate dataset untuk setiap observable secara paralel
    - Train model dari aggregated data
    - Check status sebelum workflow B
    
    Parameters
    ----------
    observable_list : List[str]
        List of observable Pauli strings
    n_circuits_per_obs : int
        Circuits per observable
    n_qubits : int
        Number of qubits
    model_id : int
        Model identifier
    threshold_r2 : float
        Threshold untuk passing workflow A
    
    Returns
    -------
    Dict
        Hasil workflow A + status decision
    """
    logger.info("=" * 70)
    logger.info("WORKFLOW A: PIPELINE DENGAN OBSERVABLE VARIATIONS")
    logger.info("=" * 70)
    
    # Task 1: Generate data untuk setiap observable (PARALEL)
    observable_tasks = []
    for obs in observable_list:
        task_result = prepare_dataset_task(
            n_circuits=n_circuits_per_obs,
            n_qubits=n_qubits,
            observable=obs
        )
        observable_tasks.append(task_result)
    
    # Task 2: Train model dari aggregated data
    training_results = train_model_task(
        observable_results=observable_tasks,
        n_qubits=n_qubits,
        model_id=model_id
    )
    
    # Task 3: Check status (dependency management)
    status = check_workflow_a_status(
        training_results=training_results,
        threshold_r2=threshold_r2
    )
    
    return {
        "workflow": "A",
        "observable_results": observable_tasks,
        "training_metrics": training_results,
        "status": status,
        "n_observables": len(observable_list),
        "n_circuits": n_circuits_per_obs * len(observable_list)
    }


# ============================================================================
# WORKFLOW B: VQE UNTUK RANGE BOND LENGTHS (PARALEL)
# ============================================================================

@ct.electron(executor=ct.executor.LocalExecutor())
def vqe_single_bondlength_task(
    bond_length: float,
    model_path: Optional[str] = None,
    n_qubits: int = 4,
    maxiter: int = 50
) -> Dict:
    """
    Electron: Run VQE untuk satu bond length.
    
    Parameters
    ----------
    bond_length : float
        H2 bond length (Angstrom)
    model_path : Optional[str]
        Path ke ML model untuk mitigation
    n_qubits : int
        Number of qubits
    maxiter : int
        Max VQE iterations
    
    Returns
    -------
    Dict
        VQE results untuk bond length ini
    """
    logger.info(f"Running VQE for bond_length={bond_length:.3f}Å")
    
    from old.qem_main_updated_v2 import H2HamiltonianGenerator, GenerateAnsatz, AnsatzConfig, VQERunner
    
    # Generate H2 Hamiltonian
    h2_gen = H2HamiltonianGenerator(n_qubits=n_qubits)
    hamiltonian = h2_gen.generate_h2_hamiltonian(bond_length)
    
    # Create ansatz
    config = AnsatzConfig(n_qubits=n_qubits)
    ansatz_gen = GenerateAnsatz(config)
    ansatz = ansatz_gen.create_ansatz()
    
    # Run VQE
    vqe_runner = VQERunner(
        n_qubits=n_qubits,
        maxiter=maxiter,
        ml_model_path=model_path
    )
    
    results = vqe_runner.run_all(ansatz, hamiltonian)
    
    # Extract energies
    result_summary = {
        "bond_length": bond_length,
        "ideal_energy": results["ideal"]["energy"],
        "noisy_energy": results["noisy"].get("energy"),
        "zne_energy": results["zne"].get("energy"),
    }
    
    if results["mitigated"].get("status") != "skipped":
        result_summary["mitigated_energy"] = results["mitigated"].get("energy")
    
    logger.info(f"✓ VQE completed for r={bond_length:.3f}Å")
    logger.info(f"  Ideal: {result_summary['ideal_energy']:.6f}")
    logger.info(f"  Noisy: {result_summary['noisy_energy']:.6f}")
    
    return result_summary


@ct.electron(executor=ct.executor.LocalExecutor())
def collect_bond_length_results(
    vqe_results: List[Dict]
) -> Dict:
    """
    Electron: Collect dan aggregate results dari semua bond lengths.
    
    Parameters
    ----------
    vqe_results : List[Dict]
        VQE results dari semua bond lengths
    
    Returns
    -------
    Dict
        Aggregated results dengan statistics
    """
    logger.info(f"Collecting results from {len(vqe_results)} bond lengths")
    
    bond_lengths = np.array([r["bond_length"] for r in vqe_results])
    ideal_energies = np.array([r["ideal_energy"] for r in vqe_results])
    noisy_energies = np.array([r["noisy_energy"] for r in vqe_results])
    zne_energies = np.array([r["zne_energy"] for r in vqe_results])
    
    # Calculate statistics
    noisy_errors = np.abs(noisy_energies - ideal_energies)
    zne_errors = np.abs(zne_energies - ideal_energies)
    
    improvement = np.mean(noisy_errors - zne_errors)
    
    return {
        "n_bond_lengths": len(vqe_results),
        "bond_lengths": bond_lengths.tolist(),
        "ideal_energies": ideal_energies.tolist(),
        "noisy_energies": noisy_energies.tolist(),
        "zne_energies": zne_energies.tolist(),
        "noisy_errors": noisy_errors.tolist(),
        "zne_errors": zne_errors.tolist(),
        "mean_noisy_error": float(np.mean(noisy_errors)),
        "mean_zne_error": float(np.mean(zne_errors)),
        "improvement": float(improvement),
        "improvement_percentage": float(improvement / np.mean(noisy_errors) * 100) if np.mean(noisy_errors) > 0 else 0
    }


@ct.lattice
def workflow_b_vqe_bondlength_range(
    bond_lengths: List[float],
    model_path: Optional[str] = None,
    n_qubits: int = 4,
    maxiter: int = 50
) -> Dict:
    """
    Workflow B: VQE untuk range bond lengths (PARALEL).
    
    Fitur:
    - Run VQE untuk setiap bond length secara paralel
    - Collect dan aggregate results
    - Compare performance across bond lengths
    
    Parameters
    ----------
    bond_lengths : List[float]
        Range of bond lengths untuk H2
    model_path : Optional[str]
        Path ke trained ML model (dari workflow A)
    n_qubits : int
        Number of qubits
    maxiter : int
        Max VQE iterations
    
    Returns
    -------
    Dict
        Hasil workflow B dengan comparison
    """
    logger.info("=" * 70)
    logger.info("WORKFLOW B: VQE UNTUK RANGE BOND LENGTHS")
    logger.info("=" * 70)
    logger.info(f"Bond lengths: {bond_lengths}")
    
    # Task 1: Run VQE untuk setiap bond length (PARALEL)
    vqe_tasks = []
    for bond_length in bond_lengths:
        task_result = vqe_single_bondlength_task(
            bond_length=bond_length,
            model_path=model_path,
            n_qubits=n_qubits,
            maxiter=maxiter
        )
        vqe_tasks.append(task_result)
    
    # Task 2: Collect dan aggregate results
    aggregated = collect_bond_length_results(vqe_results=vqe_tasks)
    
    return {
        "workflow": "B",
        "individual_results": vqe_tasks,
        "aggregated": aggregated,
        "n_bond_lengths": len(bond_lengths),
        "improvement_percent": aggregated["improvement_percentage"]
    }


# ============================================================================
# MASTER WORKFLOW: WORKFLOW A + B DENGAN DEPENDENCY
# ============================================================================

@ct.lattice
def master_workflow(
    observable_list: List[str] = None,
    n_circuits_per_obs: int = 500,
    bond_lengths: List[float] = None,
    n_qubits: int = 4,
    model_id: int = 1,
    maxiter: int = 50,
    threshold_r2: float = 0.8,
    skip_workflow_a: bool = False,
    model_path: Optional[str] = None,
) -> Dict:
    """
    Master Workflow: Orchestrate workflows A dan B dengan dependency management.
    
    DEPENDENCIES:
    - Workflow A harus pass check sebelum Workflow B dimulai
    - Workflow B menggunakan model dari Workflow A
    
    Parameters
    ----------
    observable_list : List[str]
        Observables untuk workflow A
    n_circuits_per_obs : int
        Circuits per observable
    bond_lengths : List[float]
        Bond lengths untuk workflow B
    n_qubits : int
        Number of qubits
    model_id : int
        Model identifier
    maxiter : int
        Max VQE iterations untuk workflow B
    threshold_r2 : float
        Threshold untuk pass workflow A
    skip_workflow_a : bool
        Skip workflow A (gunakan model existing)
    model_path : Optional[str]
        Path ke existing model (jika skip_workflow_a=True)
    
    Returns
    -------
    Dict
        Results dari workflow A dan B
    """
    logger.info("=" * 70)
    logger.info("MASTER WORKFLOW: ML-QEM PIPELINE + VQE BOND LENGTH SCAN")
    logger.info("=" * 70)
    
    if observable_list is None:
        observable_list = ["IIIZ", "IIZI", "IZII", "ZIII"]
    
    if bond_lengths is None:
        bond_lengths = np.linspace(0.4, 2.0, 9).tolist()
    
    # ========== WORKFLOW A ==========
    if skip_workflow_a:
        logger.info("SKIPPING WORKFLOW A (using existing model)")
        workflow_a_result = {
            "status": {"passed": True, "model_path": model_path},
            "skipped": True
        }
    else:
        logger.info("STARTING WORKFLOW A: Training ML Model...")
        workflow_a_result = workflow_a_pipeline(
            observable_list=observable_list,
            n_circuits_per_obs=n_circuits_per_obs,
            n_qubits=n_qubits,
            model_id=model_id,
            threshold_r2=threshold_r2
        )
    
    # ========== DEPENDENCY CHECK ==========
    workflow_a_passed = workflow_a_result["status"]["passed"]
    trained_model_path = workflow_a_result["status"].get("model_path") or model_path
    
    if not workflow_a_passed:
        logger.error(f"❌ WORKFLOW A FAILED - Halting pipeline")
        logger.error(f"   R² threshold ({threshold_r2}) not met")
        return {
            "status": "FAILED",
            "reason": "Workflow A did not pass quality check",
            "workflow_a": workflow_a_result,
            "workflow_b": None
        }
    
    logger.info("✓ WORKFLOW A PASSED - Proceeding with Workflow B")
    
    # ========== WORKFLOW B ==========
    logger.info("STARTING WORKFLOW B: VQE Bond Length Scan...")
    workflow_b_result = workflow_b_vqe_bondlength_range(
        bond_lengths=bond_lengths,
        model_path=trained_model_path,
        n_qubits=n_qubits,
        maxiter=maxiter
    )
    
    # ========== FINAL SUMMARY ==========
    final_result = {
        "status": "COMPLETED",
        "workflow_a": workflow_a_result,
        "workflow_b": workflow_b_result,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_observables": len(observable_list),
            "total_circuits": n_circuits_per_obs * len(observable_list),
            "total_bond_lengths": len(bond_lengths),
            "model_path": trained_model_path,
            "improvement_percent": workflow_b_result["improvement_percent"]
        }
    }
    
    return final_result


# ============================================================================
# EXECUTION UTILITIES
# ============================================================================

def run_master_workflow_local(
    observable_list: List[str] = None,
    n_circuits_per_obs: int = 100,  # reduced for testing
    bond_lengths: List[float] = None,
    n_qubits: int = 4,
    model_id: int = 1,
    maxiter: int = 25,
    output_dir: str = "workflow_output"
) -> Dict:
    """
    Run master workflow dengan local executor.
    
    Parameters
    ----------
    observable_list : List[str]
        List of observables
    n_circuits_per_obs : int
        Circuits per observable
    bond_lengths : List[float]
        Bond lengths untuk VQE
    n_qubits : int
        Number of qubits
    model_id : int
        Model identifier
    maxiter : int
        Max VQE iterations
    output_dir : str
        Output directory untuk hasil
    
    Returns
    -------
    Dict
        Workflow results
    """
    if not COVALENT_AVAILABLE:
        raise RuntimeError("Covalent not installed. Install with: pip install covalent")
    
    logger.info("Running master workflow with LOCAL executor...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dispatch workflow
    dispatch_id = ct.dispatch(master_workflow)(
        observable_list=observable_list,
        n_circuits_per_obs=n_circuits_per_obs,
        bond_lengths=bond_lengths,
        n_qubits=n_qubits,
        model_id=model_id,
        maxiter=maxiter,
        threshold_r2=0.8,
        skip_workflow_a=False,
        model_path=None
    )
    
    # Get results
    logger.info(f"Workflow dispatched with ID: {dispatch_id}")
    result = ct.get_result(dispatch_id, wait=True)
    
    # Save results
    result_file = os.path.join(output_dir, f"workflow_result_{dispatch_id}.json")
    with open(result_file, "w") as f:
        json.dump(result.result, f, indent=2)
    
    logger.info(f"✓ Results saved to: {result_file}")
    
    return result.result


def run_master_workflow_remote(
    observable_list: List[str] = None,
    n_circuits_per_obs: int = 100,
    bond_lengths: List[float] = None,
    n_qubits: int = 4,
    model_id: int = 1,
    maxiter: int = 25,
    slurm_config: Dict = None,
    output_dir: str = "workflow_output"
) -> Dict:
    """
    Run master workflow dengan remote SLURM executor.
    
    Parameters
    ----------
    observable_list : List[str]
        List of observables
    n_circuits_per_obs : int
        Circuits per observable
    bond_lengths : List[float]
        Bond lengths untuk VQE
    n_qubits : int
        Number of qubits
    model_id : int
        Model identifier
    maxiter : int
        Max VQE iterations
    slurm_config : Dict
        SLURM configuration
    output_dir : str
        Output directory untuk hasil
    
    Returns
    -------
    Dict
        Workflow results
    """
    if not COVALENT_AVAILABLE:
        raise RuntimeError("Covalent not installed")
    
    logger.info("Running master workflow with REMOTE (SLURM) executor...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if slurm_config is None:
        slurm_config = {}
    
    # Dispatch workflow ke SLURM
    dispatch_id = ct.dispatch(master_workflow)(
        observable_list=observable_list,
        n_circuits_per_obs=n_circuits_per_obs,
        bond_lengths=bond_lengths,
        n_qubits=n_qubits,
        model_id=model_id,
        maxiter=maxiter,
        threshold_r2=0.8,
        skip_workflow_a=False,
        model_path=None
    )
    
    logger.info(f"Workflow dispatched with ID: {dispatch_id}")
    logger.info("Check status with: covalent logs [dispatch_id]")
    
    # Optional: wait for results
    result = ct.get_result(dispatch_id, wait=True)
    
    # Save results
    result_file = os.path.join(output_dir, f"workflow_result_{dispatch_id}.json")
    with open(result_file, "w") as f:
        json.dump(result.result, f, indent=2)
    
    logger.info(f"✓ Results saved to: {result_file}")
    
    return result.result


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Covalent Workflow Orchestration untuk ML-QEM Pipeline"
    )
    
    parser.add_argument(
        "--executor",
        type=str,
        default="local",
        choices=["local", "remote"],
        help="Executor type"
    )
    
    parser.add_argument(
        "--n-circuits",
        type=int,
        default=100,
        help="Circuits per observable"
    )
    
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=4,
        help="Number of qubits"
    )
    
    parser.add_argument(
        "--maxiter",
        type=int,
        default=25,
        help="Max VQE iterations"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="workflow_output",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Define observables dan bond lengths
    observables = ["IIIZ", "IIZI", "IZII"]
    bond_lengths = [0.5, 0.74, 1.0, 1.5, 2.0]
    
    if args.executor == "local":
        results = run_master_workflow_local(
            observable_list=observables,
            n_circuits_per_obs=args.n_circuits,
            bond_lengths=bond_lengths,
            n_qubits=args.n_qubits,
            maxiter=args.maxiter,
            output_dir=args.output_dir
        )
    else:
        results = run_master_workflow_remote(
            observable_list=observables,
            n_circuits_per_obs=args.n_circuits,
            bond_lengths=bond_lengths,
            n_qubits=args.n_qubits,
            maxiter=args.maxiter,
            output_dir=args.output_dir
        )
    
    # Print summary
    print("\n" + "=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print(json.dumps(results.get("summary", {}), indent=2))
