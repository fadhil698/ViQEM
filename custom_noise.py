import numpy as np
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError

class CustomNoiseBuilder:
    def __init__(self, t1=50e3, t2=70e3, gate_time_1q=50, gate_time_2q=300,
                 p_depol_1q=0.001, 
                 p_depol_2q=0.01,   # REVISI 3: Tambah Error 2-Qubit (1%)
                 p_readout=0.03):
        """
        Parameter disesuaikan agar mimic 'Real Hardware' (Athens/Belem):
        - Gate 1Q error: 0.1%
        - Gate 2Q error: 1.0% (CNOT jauh lebih berisik di hardware asli)
        - Readout error: 3.0%
        """
        self.t1 = t1
        self.t2 = t2
        self.gate_time_1q = gate_time_1q
        self.gate_time_2q = gate_time_2q
        self.p_depol_1q = p_depol_1q
        self.p_depol_2q = p_depol_2q
        self.p_readout = p_readout

        # Pre-build error objects
        self._error_1q = self._build_error_1q()
        self._error_2q = self._build_error_2q()
        self._error_readout = self._build_readout_error()

    def _build_error_1q(self):
        thermal = thermal_relaxation_error(self.t1, self.t2, self.gate_time_1q)
        depol = depolarizing_error(self.p_depol_1q, 1)
        return depol.compose(thermal)

    def _build_error_2q(self):
        # Untuk gate 2-qubit, depolarizing error 2-qubit adalah model yang paling 
        # umum dan secara analitik mudah dilacak (damping factor 1-4/3p).
        return depolarizing_error(self.p_depol_2q, 2)

    def _build_readout_error(self):
        p = self.p_readout
        prob_matrix = [[1 - p, p], [p, 1 - p]]
        return ReadoutError(prob_matrix)

    def get_noise_model(self, num_qubits=4, target_qubits=None):
        """
        num_qubits: jumlah total qubit dalam sirkuit (2 atau 4)
        faulty_qubits: list qubit yang 'ekstra rusak' gate-nya
        """
        noise_model = NoiseModel()
        all_qubits = list(range(num_qubits))
        gates_1q = ["u1", "u2", "u3", "rz", "sx", "x", "h", "ry", "id"]

        # 1. BERIKAN READOUT ERROR KE SEMUA QUBIT (Global Noise)
        for q in all_qubits:
            noise_model.add_readout_error(self._error_readout, [q])
            
            # Opsional: Berikan gate error 1-Q kecil sebagai background noise
            # agar sirkuit tidak terlalu 'sempurna' di qubit lain
            noise_model.add_quantum_error(self._error_1q, gates_1q, [q])

        # 2. BERIKAN CNOT ERROR (Linear Entanglement)
        # Sesuai ansatz kamu: [0,1], [1,2], [2,3]
        for q1 in range(num_qubits - 1):
            q2 = q1 + 1
            noise_model.add_quantum_error(self._error_2q, ["cx"], [q1, q2])
            noise_model.add_quantum_error(self._error_2q, ["cx"], [q2, q1])

        # 3. BERIKAN EXTRA ERROR PADA FAULTY QUBIT (Jika ada)
        # Kamu bisa membuat error yang lebih berat di sini jika ingin 
        # mensimulasikan qubit yang benar-benar 'sakit'
        if target_qubits:
            heavy_error_1q = depolarizing_error(0.005, 1) # Lebih besar dari p_depol_1q
            for q in target_qubits:
                noise_model.add_quantum_error(heavy_error_1q, gates_1q, [q])

        return noise_model