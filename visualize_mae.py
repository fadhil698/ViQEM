import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MLEvalVisualizer:
    def __init__(self, base_path, default_fingerprint=""):
        self.base_path = base_path
        self.fingerprint = default_fingerprint

    def load_data(self, mapper, grouping, exp_id, specific_fingerprint=None):
        """Load data dari ml_evaluation dan hitung Absolute Error (AE)."""
        fp = specific_fingerprint if specific_fingerprint else self.fingerprint
        
        # Sesuai instruksi, kita arahkan ke folder ml_evaluation
        path = os.path.join(
            self.base_path, f"out_{fp}", f"unified_output_{mapper}", 
            grouping, exp_id, "ml_evaluation", "predictions_1.csv"
        )
        
        # Fallback jika ternyata filenya ada di predict_data (seperti di snippet-mu)
        if not os.path.exists(path):
            path = os.path.join(
                self.base_path, f"out_{fp}", f"unified_output_{mapper}", 
                grouping, exp_id, "ml_training", "predict_data", "predicted_data_1.csv"
            )
            
        if not os.path.exists(path):
            return None
        
        df = pd.read_csv(path)
        
        # Fleksibilitas nama kolom (RF_energy vs mitigated_energy)
        col_rf = 'RF_energy' if 'RF_energy' in df.columns else 'mitigated_energy'
        
        # Hitung Absolute Error
        if 'noisy_energy' in df.columns and 'ideal_energy' in df.columns:
            df['AE_Noisy'] = (df['noisy_energy'] - df['ideal_energy']).abs()
        if 'zne_energy' in df.columns and 'ideal_energy' in df.columns:
            df['AE_ZNE'] = (df['zne_energy'] - df['ideal_energy']).abs()
        if col_rf in df.columns and 'ideal_energy' in df.columns:
            df['AE_RF'] = (df[col_rf] - df['ideal_energy']).abs()
            
        return df

    def plot_evaluation_bars(self, experiment_list, fingerprints=None, 
                             show_methods=['Noisy', 'ZNE', 'RF'], 
                             save_name="ml_eval_bars.png", log_scale=False):
        """
        Membuat Grouped Bar Chart untuk membandingkan MAE antar eksperimen.
        Jika fingerprints diberikan dalam bentuk list, akan menghitung statistik gabungan.
        """
        # Gunakan single fingerprint jika list tidak diberikan
        fps_to_process = fingerprints if fingerprints else [self.fingerprint]
        
        # 1. Kelompokkan eksperimen per mapper
        grouped_exps = {}
        for exp in experiment_list:
            m = exp['mapper']
            if m not in grouped_exps: grouped_exps[m] = []
            grouped_exps[m].append(exp)

        # Siapkan list untuk tabel DataFrame
        stats_list = []
        
        n_mappers = len(grouped_exps)
        fig, axes = plt.subplots(1, n_mappers, figsize=(8*n_mappers, 6), squeeze=False)
        
        # Konfigurasi Warna
        color_map = {'Noisy': 'salmon', 'ZNE': 'silver', 'RF': 'royalblue'}
        
        for i, (mapper, exps) in enumerate(grouped_exps.items()):
            ax = axes[0, i]
            
            # Posisi bar
            x = np.arange(len(exps))
            width = 0.8 / len(show_methods) # Lebar dinamis
            
            # Dictionary untuk menyimpan data plot
            plot_data = {m: {'means': [], 'stds': []} for m in show_methods}
            exp_labels = []
            
            # Kumpulkan data statistik
            for exp in exps:
                exp_labels.append(f"{exp['id']}\n({exp['grouping']})")
                all_data = []
                
                for fp in fps_to_process:
                    df = self.load_data(exp['mapper'], exp['grouping'], exp['id'], specific_fingerprint=fp)
                    if df is not None:
                        all_data.append(df)
                
                if not all_data:
                    for m in show_methods:
                        plot_data[m]['means'].append(0)
                        plot_data[m]['stds'].append(0)
                    continue
                
                # Gabungkan data untuk statistik yang lebih solid
                combined_df = pd.concat(all_data)
                
                # Simpan statistik untuk tabel summary
                stat_row = {'Mapper': mapper.upper(), 'Experiment': exp['id'], 'Grouping': exp['grouping']}
                
                for method in show_methods:
                    col_name = f'AE_{method}'
                    if col_name in combined_df.columns:
                        mean_val = combined_df[col_name].mean()
                        std_val = combined_df[col_name].std()
                    else:
                        mean_val, std_val = np.nan, np.nan
                        
                    plot_data[method]['means'].append(mean_val)
                    plot_data[method]['stds'].append(std_val)
                    
                    stat_row[f'{method} Mean'] = mean_val
                    stat_row[f'{method} Std'] = std_val
                
                stats_list.append(stat_row)

            # --- PLOTTING BAR ---
            for j, method in enumerate(show_methods):
                offset = (j - len(show_methods)/2 + 0.5) * width
                bars = ax.bar(x + offset, plot_data[method]['means'], width, 
                              yerr=plot_data[method]['stds'], label=method, 
                              color=color_map.get(method, 'gray'), alpha=0.85, 
                              edgecolor='black', capsize=5)
                
                # Tambahkan angka di atas bar
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height) and height > 0:
                        # Format lebih pendek jika log scale
                        lbl = f'{height:.4f}' if not log_scale else f'{height:.1e}'
                        ax.text(bar.get_x() + bar.get_width()/2., height, lbl,
                                ha='center', va='bottom', fontsize=9, rotation=90, padding=3)

            # Kosmetik Subplot
            ax.set_title(f"Mapper: {mapper.upper()}", fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(exp_labels, fontsize=11, fontweight='bold')
            if log_scale:
                ax.set_yscale('log')
                ax.axhline(y=0.0016, color='black', linestyle='-.', alpha=0.5, label='Chem. Accuracy')
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            
            if i == 0:
                ax.set_ylabel("Mean Absolute Error (Hartree)", fontsize=12)
            if i == n_mappers - 1:
                ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ Bar chart disimpan sebagai: {save_name}")
        plt.show()

        # Mengembalikan DataFrame agar bisa diprint/disimpan
        df_stats = pd.DataFrame(stats_list)
        return df_stats

# =====================================================================
# CARA PENGGUNAAN (MAIN EXECUTION)
# =====================================================================
if __name__ == "__main__":
    # Inisiasi dengan path utama
    viz = MLEvalVisualizer(base_path="./", default_fingerprint="49d3cb314b5c")

    # Daftar eksperimen yang ingin dievaluasi performa model ML-nya
    target_list = [
        {'mapper': 'jw', 'grouping': 'all',    'id': 'J-A-1'},
        {'mapper': 'jw', 'grouping': 'single', 'id': 'J-S-1'},
        {'mapper': 'bk', 'grouping': 'all',    'id': 'BK-A-1'},
    ]

    print("--- Membuat Plot Evaluasi ML (Mode 1: Single Fingerprint) ---")
    df_summary = viz.plot_evaluation_bars(
        experiment_list=target_list,
        show_methods=['Noisy', 'ZNE', 'RF'],
        save_name="evaluasi_ml_model.png",
        log_scale=False # Ubah ke True jika MAE Noisy dan RF bedanya sangat jauh (misal 1 vs 0.001)
    )

    print("\n--- Tabel Ringkasan (Seperti di kode aslimu) ---")
    pd.options.display.float_format = '{:,.5f}'.format
    print(df_summary.to_string(index=False))
