import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MLEvalUnifiedVisualizer:
    def __init__(self, base_path, default_fingerprint=""):
        self.base_path = base_path
        self.fingerprint = default_fingerprint
        
        # --- MAPPING DICTIONARY PAULI STRINGS ---
        self.pauli_maps = {
            'jw': {
                '1': 'IIII',  '2': 'IIIZ',  '3': 'IIZI',  '4': 'IIZZ',
                '5': 'IZII',  '6': 'IZIZ',  '7': 'ZIII',  '8': 'ZIIZ',
                '9': 'YYYY',  '10': 'XXYY', '11': 'YYXX', '12': 'XXXX',
                '13': 'IZZI', '14': 'ZIZI', '15': 'ZZII'
            },
            'bk': {
                '1': 'IIII',  '2': 'IIIZ',  '3': 'IIZZ',  '4': 'IIZI',
                '5': 'IZII',  '6': 'IZIZ',  '7': 'ZZZI',  '8': 'ZZZZ',
                '9': 'ZXIX',  '10': 'IXZX', '11': 'ZXZX', '12': 'IXIX',
                '13': 'IZZZ', '14': 'ZZIZ', '15': 'ZIZI'
            },
            'parity': {
                '1': 'II', '2': 'IZ', '3': 'ZI', '4': 'ZZ', '5': 'XX'
            }
        }

    # --- UTILITY FUNCTIONS ---
    def natural_key(self, text):
        return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

    def get_pauli_label(self, mapper, dataset_name):
        match = re.search(r'-(\d+)', dataset_name)
        if match:
            num_id = match.group(1)
            return self.pauli_maps.get(mapper, {}).get(num_id, dataset_name)
        return dataset_name

    def load_data(self, mapper, grouping, exp_id, specific_fingerprint=None):
        fp = specific_fingerprint if specific_fingerprint else self.fingerprint
        
        # Prioritas 1: ml_evaluation
        path = os.path.join(
            self.base_path, f"out_{fp}", f"unified_output_{mapper}", 
            grouping, exp_id, "ml_evaluation", "predictions_1.csv"
        )
        
        # Prioritas 2: ml_training/predict_data (Fallback)
        if not os.path.exists(path):
            path = os.path.join(
                self.base_path, f"out_{fp}", f"unified_output_{mapper}", 
                grouping, exp_id, "ml_training", "predict_data", "predicted_data_1.csv"
            )
            
        if not os.path.exists(path): 
            return None
        
        df = pd.read_csv(path)
        col_rf = 'RF_energy' if 'RF_energy' in df.columns else 'mitigated_energy'
        
        # Hitung Error Mutlak (AE) jika datanya ada
        if 'ideal_energy' in df.columns:
            if 'noisy_energy' in df.columns: df['AE_Noisy'] = (df['noisy_energy'] - df['ideal_energy']).abs()
            if 'zne_energy' in df.columns:   df['AE_ZNE']   = (df['zne_energy'] - df['ideal_energy']).abs()
            if col_rf in df.columns:         df['AE_RF']    = (df[col_rf] - df['ideal_energy']).abs()
            return df
        return None

    # =====================================================================
    # MODE 1: KOMPARASI METODE (NOISY vs ZNE vs RF) & TABEL STATISTIK
    # =====================================================================
    def plot_evaluation_bars(self, experiment_list, fingerprints=None, 
                             show_methods=['Noisy', 'ZNE', 'RF'], 
                             save_name="ml_eval_bars.png", log_scale=False):
        fps_to_process = fingerprints if fingerprints else [self.fingerprint]
        
        grouped_exps = {}
        for exp in experiment_list:
            m = exp['mapper']
            if m not in grouped_exps: grouped_exps[m] = []
            grouped_exps[m].append(exp)

        stats_list = []
        n_mappers = len(grouped_exps)
        fig, axes = plt.subplots(1, n_mappers, figsize=(8*n_mappers, 6), squeeze=False)
        color_map = {'Noisy': 'salmon', 'ZNE': 'silver', 'RF': 'royalblue'}
        
        for i, (mapper, exps) in enumerate(grouped_exps.items()):
            ax = axes[0, i]
            x = np.arange(len(exps))
            width = 0.8 / len(show_methods)
            
            plot_data = {m: {'means': [], 'stds': []} for m in show_methods}
            exp_labels = []
            
            for exp in exps:
                exp_labels.append(f"{exp['id']}\n({exp['grouping']})")
                all_data = []
                
                for fp in fps_to_process:
                    df = self.load_data(exp['mapper'], exp['grouping'], exp['id'], specific_fingerprint=fp)
                    if df is not None: all_data.append(df)
                
                if not all_data:
                    for m in show_methods:
                        plot_data[m]['means'].append(0); plot_data[m]['stds'].append(0)
                    continue
                
                combined_df = pd.concat(all_data)
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

            for j, method in enumerate(show_methods):
                offset = (j - len(show_methods)/2 + 0.5) * width
                bars = ax.bar(x + offset, plot_data[method]['means'], width, 
                              yerr=plot_data[method]['stds'], label=method, 
                              color=color_map.get(method, 'gray'), alpha=0.85, 
                              edgecolor='black', capsize=5)
                
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height) and height > 0:
                        lbl = f'{height:.4f}' if not log_scale else f'{height:.1e}'
                        ax.text(bar.get_x() + bar.get_width()/2., height, lbl,
                                ha='center', va='bottom', fontsize=9, rotation=90, padding=3)

            ax.set_title(f"Mapper: {mapper.upper()}", fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(exp_labels, fontsize=11, fontweight='bold')
            if log_scale:
                ax.set_yscale('log')
                ax.axhline(y=0.0016, color='black', linestyle='-.', alpha=0.5, label='Chem. Accuracy')
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            
            if i == 0: ax.set_ylabel("Mean Absolute Error (Hartree)", fontsize=12)
            if i == n_mappers - 1: ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ Plot Komparasi Metode disimpan: {save_name}")
        plt.show()

        return pd.DataFrame(stats_list)

    # =====================================================================
    # MODE 2: KOMPARASI KARDINALITAS & PAULI STRING (FITUR BARU)
    # =====================================================================
    def plot_pauli_comparison(self, mapper, target_experiments, fingerprints=None, save_name="pauli_comparison.png"):
        fps_to_process = fingerprints if fingerprints else [self.fingerprint]
        all_data = []
        
        for exp in target_experiments:
            cardinality = exp['grouping'].capitalize()
            pauli_lbl = self.get_pauli_label(mapper, exp['id'])
            
            for fp in fps_to_process:
                df = self.load_data(mapper, exp['grouping'], exp['id'], specific_fingerprint=fp)
                if df is not None and 'AE_RF' in df.columns:
                    mae = df['AE_RF'].mean()
                    all_data.append({
                        'cardinality': cardinality,
                        'dataset_folder': exp['id'],
                        'pauli_label': pauli_lbl,
                        'mae': mae
                    })
                    
        df_summary = pd.DataFrame(all_data)
        if not df_summary.empty:
            df_summary = df_summary[~df_summary['dataset_folder'].str.contains('-1$')] 

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 9), sharey=False)
        sns.set_style("whitegrid")
        card_colors = {'Single': '#4C72B0', 'Dual': '#55A868', 'Triple': '#C44E52'} 

        for i, card in enumerate(['Single', 'Dual', 'Triple']):
            ax = axes[i]
            df_subset = df_summary[df_summary['cardinality'] == card] if not df_summary.empty else pd.DataFrame()
            
            if df_subset.empty:
                ax.set_title(f"Data tidak ditemukan\n({card})", fontsize=16)
                continue

            unique_datasets = sorted(df_subset['dataset_folder'].unique(), key=self.natural_key)
            order_paulis = [self.get_pauli_label(mapper, ds) for ds in unique_datasets]

            sns.barplot(
                data=df_subset, x='pauli_label', y='mae',
                order=order_paulis, color=card_colors.get(card, 'gray'), 
                linewidth=1.5, ax=ax, errorbar=None 
            )

            for spine in ax.spines.values():
                spine.set_visible(True)      
                spine.set_color('black')     
                spine.set_linewidth(1.5)     

            for container in ax.containers:
                values = [bar.get_height() for bar in container]
                valid_values = [v for v in values if not np.isnan(v) and v > 0]
                if not valid_values: continue
                min_val = min(valid_values)
                
                for bar in container:
                    height = bar.get_height()
                    if np.isnan(height) or height <= 0: continue
                    if abs(height - min_val) < 1e-9:
                        ax.text(
                            bar.get_x() + bar.get_width()/2, height, 
                            '★', ha='center', va='bottom', color='red', fontsize=30, fontweight='bold'
                        )

            ax.set_title(f"{card}", fontsize=18, fontweight='bold', pad=15)
            ax.set_xlabel("Pauli String / Kombinasi", fontsize=16, labelpad=10)
            if i == 0: ax.set_ylabel("Mean Absolute Error (RF)", fontsize=16)
            else: ax.set_ylabel("") 
                
            ax.tick_params(axis='y', labelsize=14)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15) 
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ Plot perbandingan Pauli disimpan: {save_name}")
        plt.show()

# =====================================================================
# CARA PENGGUNAAN (MAIN EXECUTION)
# =====================================================================
if __name__ == "__main__":
    # 1. INISIASI VISUALIZER
    viz = MLEvalUnifiedVisualizer(base_path="./", default_fingerprint="49d3cb314b5c")

    # Siapkan beberapa list fingerprint jika ingin melakukan analisis statistik (opsional)
    my_fingerprints = ["49d3cb314b5c"] # Tambahkan fingerprint lain ke list ini jika ada

    # =================================================================
    # MENJALANKAN MODE 1: KOMPARASI METODE (Noisy vs ZNE vs RF)
    # =================================================================
    print("\n" + "="*50)
    print("Mengeksekusi MODE 1: Komparasi Performa Keseluruhan")
    print("="*50)
    
    exp_list_mode_1 = [
        {'mapper': 'jw', 'grouping': 'all',    'id': 'J-A-1'},
        {'mapper': 'jw', 'grouping': 'single', 'id': 'J-S-1'},
        {'mapper': 'bk', 'grouping': 'all',    'id': 'BK-A-1'},
    ]
    
    df_stats = viz.plot_evaluation_bars(
        experiment_list=exp_list_mode_1,
        fingerprints=my_fingerprints,
        show_methods=['Noisy', 'ZNE', 'RF'],
        save_name="graph/eval_mode1_methods.png",
        log_scale=False
    )
    
    # Print tabel statistik
    pd.options.display.float_format = '{:,.5f}'.format
    print("\n--- Tabel Statistik Evaluasi ---")
    print(df_stats.to_string(index=False))

    # =================================================================
    # MENJALANKAN MODE 2: KOMPARASI PAULI STRINGS (Single, Dual, Triple)
    # =================================================================
    print("\n" + "="*50)
    print("Mengeksekusi MODE 2: Komparasi Pauli Berdasarkan Kardinalitas")
    print("="*50)
    
    exp_list_mode_2 = []
    auto_exp_list = []
    for i in range(2, 16): auto_exp_list.append({'mapper': 'jw', 'grouping': 'single', 'id': f'J-S-{i}'})
    for i in range(2, 11): auto_exp_list.append({'mapper': 'jw', 'grouping': 'dual', 'id': f'J-D-{i}'})
    for i in range(2, 6):  auto_exp_list.append({'mapper': 'jw', 'grouping': 'triple', 'id': f'J-T-{i}'})
    
    viz.plot_pauli_comparison(
        mapper='jw', # Pastikan mapper yang dipilih sesuai dengan eksperimennya
        target_experiments=auto_exp_list,
        fingerprints=my_fingerprints,
        save_name="graph/eval_mode2_pauli.png"
    )