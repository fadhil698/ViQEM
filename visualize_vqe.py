import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit

class VQEUnifiedVisualizer:
    def __init__(self, base_path, default_fingerprint=""):
        self.base_path = base_path
        self.fingerprint = default_fingerprint
        self.repulsion_cache = {}

    def get_nuc_repulsion(self, r):
        """Menghitung dan me-cache nilai energi nuklir berdasarkan jarak atom."""
        if r in self.repulsion_cache: 
            return self.repulsion_cache[r]
        try:
            driver = PySCFDriver(atom=f"H 0 0 0; H 0 0 {r}", basis="sto3g", unit=DistanceUnit.ANGSTROM)
            problem = driver.run()
            energy = problem.nuclear_repulsion_energy
            self.repulsion_cache[r] = energy
            return energy
        except Exception as e: 
            print(f"Error PySCF untuk R={r}: {e}")
            return 0.0

    def load_data(self, mapper, grouping, exp_id, specific_fingerprint=None):
        """Load data CSV dan konversi ke Absolute Error."""
        # Gunakan fingerprint spesifik jika diberikan (untuk mode statistik), 
        # jika tidak, gunakan fingerprint default.
        fp = specific_fingerprint if specific_fingerprint else self.fingerprint
        
        path = os.path.join(
            self.base_path, f"out_{fp}", f"unified_output_{mapper}", 
            grouping, exp_id, "vqe_execution/vqe_results_multi_bond_length.csv"
        )
        
        if not os.path.exists(path):
            return None
        
        df = pd.read_csv(path)
        df['nuc_repulsion'] = df['bond_length'].apply(self.get_nuc_repulsion)
        
        # Hitung Absolute Error
        for col in ['ideal_energy', 'noisy_energy', 'mitigated_energy', 'zne_energy']:
            if col in df.columns:
                df[f'total_{col}'] = df[col] + df['nuc_repulsion']
                if col != 'ideal_energy':
                    df[f'err_{col}'] = (df[f'total_{col}'] - df['total_ideal_energy']).abs()
        return df

    # =====================================================================
    # FITUR 1: PERBANDINGAN BANYAK EKSPERIMEN DALAM 1 FINGERPRINT
    # =====================================================================
    def plot_smart_comparison(self, experiment_list, save_name="general_vqe_plot.png"):
        grouped_exps = {}
        for exp in experiment_list:
            m = exp['mapper']
            if m not in grouped_exps: grouped_exps[m] = []
            grouped_exps[m].append(exp)

        n_mappers = len(grouped_exps)
        fig, axes = plt.subplots(1, n_mappers, figsize=(7*n_mappers, 6), squeeze=False)

        for i, (mapper, exps) in enumerate(grouped_exps.items()):
            ax = axes[0, i]
            colors = plt.cm.tab10.colors 
            
            for j, exp in enumerate(exps):
                df = self.load_data(exp['mapper'], exp['grouping'], exp['id'])
                if df is None: continue
                
                label_base = f"{exp['id']} ({exp['grouping']})"
                color = colors[j % len(colors)]

                if 'err_noisy_energy' in df.columns:
                    ax.plot(df['bond_length'], df['err_noisy_energy'], '--', color=color, alpha=0.3)
                if 'err_mitigated_energy' in df.columns:
                    ax.plot(df['bond_length'], df['err_mitigated_energy'], '-o', 
                            color=color, label=f"{label_base} - RF", markersize=4)
                if 'err_zne_energy' in df.columns:
                    ax.plot(df['bond_length'], df['err_zne_energy'], ':s', 
                            color=color, label=f"{label_base} - ZNE", markersize=4)

            ax.axhline(y=0.0016, color='black', linestyle='-.', alpha=0.5, label='Chem. Accuracy')
            
            ax.set_yscale('log')
            ax.set_title(f"Mapper: {mapper.upper()}", fontsize=15, fontweight='bold')
            ax.set_xlabel("Bond Length (Å)", fontsize=12)
            if i == 0: ax.set_ylabel("Absolute Error (Hartree)", fontsize=12)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend(fontsize=9, loc='lower right')

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"Plot Perbandingan disimpan: {save_name}")
        plt.show()

    # =====================================================================
    # FITUR 2: ANALISIS STATISTIK DARI BANYAK FINGERPRINT (MEAN & STD DEV)
    # =====================================================================
    def plot_statistical_comparison(self, fingerprints, mapper, grouping, exp_id, save_name="vqe_stats.png"):
        all_data = []

        for fp in fingerprints:
            # Panggil load_data dengan specific_fingerprint
            df = self.load_data(mapper, grouping, exp_id, specific_fingerprint=fp)
            if df is not None:
                cols_to_keep = ['bond_length']
                if 'err_noisy_energy' in df.columns: cols_to_keep.append('err_noisy_energy')
                if 'err_mitigated_energy' in df.columns: cols_to_keep.append('err_mitigated_energy')
                
                all_data.append(df[cols_to_keep])

        if not all_data:
            print(f"Tidak ada data ditemukan untuk {mapper} - {exp_id} di list fingerprint tersebut.")
            return

        combined_df = pd.concat(all_data)
        stats = combined_df.groupby('bond_length').agg(['mean', 'std']).reset_index()

        plt.figure(figsize=(10, 6))
        
        # Plot Noisy (Mean + Area Std Dev)
        if 'err_noisy_energy' in stats.columns:
            plt.plot(stats['bond_length'], stats['err_noisy_energy']['mean'], '--', color='red', label='Mean Noisy')
            plt.fill_between(stats['bond_length'], 
                             stats['err_noisy_energy']['mean'] - stats['err_noisy_energy']['std'],
                             stats['err_noisy_energy']['mean'] + stats['err_noisy_energy']['std'], 
                             color='red', alpha=0.1)

        # Plot Mitigated (Mean + Area Std Dev)
        if 'err_mitigated_energy' in stats.columns:
            plt.plot(stats['bond_length'], stats['err_mitigated_energy']['mean'], '-', color='blue', label='Mean Mitigated (RF)', linewidth=2)
            plt.fill_between(stats['bond_length'], 
                             stats['err_mitigated_energy']['mean'] - stats['err_mitigated_energy']['std'],
                             stats['err_mitigated_energy']['mean'] + stats['err_mitigated_energy']['std'], 
                             color='blue', alpha=0.2)

        plt.axhline(y=0.0016, color='black', linestyle='-.', label='Chemical Accuracy')

        plt.yscale('log')
        plt.title(f"Statistical Robustness: {mapper.upper()} | {exp_id} ({grouping})\nAcross {len(fingerprints)} Noise Profiles (Fingerprints)", fontsize=14)
        plt.xlabel("Bond Length (Å)", fontsize=12)
        plt.ylabel("Absolute Error (Hartree)", fontsize=12)
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        print(f"Plot Statistik disimpan: {save_name}")
        plt.show()
        
    # =====================================================================
    # FITUR 3: STATISTIK MULTI-EKSPERIMEN DENGAN FILTER JENIS ENERGI
    # =====================================================================
    def plot_smart_statistical_comparison(self, fingerprints, experiment_list, 
                                          show_energies=['noisy', 'rf', 'zne'], # Parameter baru!
                                          save_name="multi_exp_stats_filtered.png"):
        grouped_exps = {}
        for exp in experiment_list:
            m = exp['mapper']
            if m not in grouped_exps: grouped_exps[m] = []
            grouped_exps[m].append(exp)

        n_mappers = len(grouped_exps)
        fig, axes = plt.subplots(1, n_mappers, figsize=(8*n_mappers, 6), squeeze=False)

        for i, (mapper, exps) in enumerate(grouped_exps.items()):
            ax = axes[0, i]
            colors = plt.cm.tab10.colors 
            
            for j, exp in enumerate(exps):
                all_data = []
                
                # Mengumpulkan data dari semua fingerprint
                for fp in fingerprints:
                    df = self.load_data(exp['mapper'], exp['grouping'], exp['id'], specific_fingerprint=fp)
                    if df is not None:
                        cols_to_keep = ['bond_length']
                        if 'err_noisy_energy' in df.columns: cols_to_keep.append('err_noisy_energy')
                        if 'err_mitigated_energy' in df.columns: cols_to_keep.append('err_mitigated_energy')
                        if 'err_zne_energy' in df.columns: cols_to_keep.append('err_zne_energy')
                        
                        all_data.append(df[cols_to_keep])
                
                if not all_data:
                    continue
                
                # Menggabungkan dan menghitung mean + std
                combined_df = pd.concat(all_data)
                stats = combined_df.groupby('bond_length').agg(['mean', 'std']).reset_index()
                
                label_base = f"{exp['id']} ({exp['grouping']})"
                color = colors[j % len(colors)]
                bond = stats['bond_length']
                
                # 1. PLOT NOISY (Hanya dieksekusi jika 'noisy' ada di show_energies)
                if 'noisy' in show_energies and 'err_noisy_energy' in stats.columns:
                    mean_val = stats['err_noisy_energy']['mean']
                    std_val  = stats['err_noisy_energy']['std']
                    ax.plot(bond, mean_val, '--', color=color, alpha=0.6, label=f"{label_base} - Noisy")
                    ax.fill_between(bond, mean_val - std_val, mean_val + std_val, color=color, alpha=0.05)

                # 2. PLOT ZNE (Hanya dieksekusi jika 'zne' ada di show_energies)
                if 'zne' in show_energies and 'err_zne_energy' in stats.columns:
                    mean_val = stats['err_zne_energy']['mean']
                    std_val  = stats['err_zne_energy']['std']
                    ax.plot(bond, mean_val, ':s', color=color, markersize=5, label=f"{label_base} - ZNE")
                    ax.fill_between(bond, mean_val - std_val, mean_val + std_val, color=color, alpha=0.1)

                # 3. PLOT MITIGATED/RF (Hanya dieksekusi jika 'rf' ada di show_energies)
                if 'rf' in show_energies and 'err_mitigated_energy' in stats.columns:
                    mean_val = stats['err_mitigated_energy']['mean']
                    std_val  = stats['err_mitigated_energy']['std']
                    ax.plot(bond, mean_val, '-o', color=color, linewidth=2, markersize=5, label=f"{label_base} - RF")
                    ax.fill_between(bond, mean_val - std_val, mean_val + std_val, color=color, alpha=0.2)

            # Garis Target (Chemical Accuracy)
            ax.axhline(y=0.0016, color='black', linestyle='-.', alpha=0.8, label='Chem. Accuracy')
            
            # Kosmetik
            ax.set_yscale('log')
            ax.set_title(f"Mapper: {mapper.upper()}", fontsize=15, fontweight='bold')
            ax.set_xlabel("Bond Length ($\AA$)", fontsize=13)
            if i == 0: ax.set_ylabel("Absolute Error (Hartree)", fontsize=13)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5)) 

        plt.tight_layout()
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
        print(f"Plot berhasil disimpan: {save_name}")
        plt.show()
# =====================================================================
# CARA PENGGUNAAN (MAIN EXECUTION)
# =====================================================================
if __name__ == "__main__":
    # Inisiasi Objek
    # Ganti "./" dengan path absolut tempat folder "out_..." berada jika perlu
    base_directory = "./" 
    
    # Masukkan fingerprint default (digunakan untuk Fitur 1)
    viz = VQEUnifiedVisualizer(base_path=base_directory, default_fingerprint="49d3cb314b5c")

    print("\n--- Menjalankan Statistik Multi-Eksperimen ---")
    
    # List dari semua fingerprint (banyak noise profile)
    list_fingerprints = [
        "49d3cb314b5c", 
        "fingerprint_kedua",   
        "fingerprint_ketiga"   
    ]
    
    # List eksperimen yang mau diadu (seperti Mode 1)
    target_list_multi = [
        {'mapper': 'jw', 'grouping': 'single', 'id': 'J-S-1'},
        {'mapper': 'jw', 'grouping': 'single', 'id': 'J-S-7'},
        {'mapper': 'bk', 'grouping': 'all',    'id': 'BK-A-1'},
        {'mapper': 'bk', 'grouping': 'single', 'id': 'BK-S-3'},
    ]
    
    viz.plot_smart_statistical_comparison(
        fingerprints=list_fingerprints,
        experiment_list=target_list_multi,
        show_energies=['noisy', 'rf', 'zne'],
        save_name="graph/vqe.png")