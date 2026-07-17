import os
import argparse
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmForSequenceClassification
import gc

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for server compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

class ProteinHalfLifeDataset(Dataset):
    """Dataset class for wild-type sequence validation inference."""
    def __init__(self, df, tokenizer, max_length=1024):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = str(row['AA'])

        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        return item

def generate_validation_predictions(args, tokenizer, folds):
    print(f"Lade normale Protein-Sequenzen von: {args.csv_path}")
    df_all = pd.read_csv(args.csv_path)
    
    # Verify split column exists
    if 'split' not in df_all.columns:
        raise KeyError("Fehler: Die Spalte 'split' fehlt in der Eingabe-CSV.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Gerät für Inferenz: {device}")

    # Load base model once
    print(f"Lade Basismodell: {args.model_name}")
    model = EsmForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=1, 
        cache_dir=args.cache_dir
    )
    model = model.to(device)
    model.eval()

    val_dfs = []

    for f_idx, splits_info in folds.items():
        val_splits = splits_info['val']
        weights_path = os.path.join(args.cache_dir, f"regression_head_weights_fold_{f_idx}.pt")
        
        if not os.path.exists(weights_path):
            print(f"Warnung: Keine Gewichte für Fold {f_idx} unter {weights_path} gefunden. Überspringe...")
            continue
            
        print(f"\n--- Generiere Validierungsvorhersagen für Fold {f_idx} ---")
        print(f"Validierungssplits: {val_splits}")
        
        # Filter sequences for validation splits of this fold
        df_val = df_all[df_all['split'].isin(val_splits)].copy().reset_index(drop=True)
        print(f"Anzahl Validierungsbeispiele: {len(df_val)}")
        
        if len(df_val) == 0:
            print("Keine Validierungsdaten für diesen Fold!")
            continue

        # Load weights into classifier
        state_dict = torch.load(weights_path, map_location=device)
        model.classifier.load_state_dict(state_dict)

        dataset = ProteinHalfLifeDataset(df_val, tokenizer)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1).cpu().numpy()
                
                if logits.ndim == 0:
                    predictions.append(float(logits))
                else:
                    predictions.extend(logits.tolist())
                
                if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
                    print(f"  Batch {batch_idx + 1}/{len(dataloader)} verarbeitet...")

        df_val['pred_halflife'] = predictions
        
        # Rename columns and keep only the specified ones
        df_val_filtered = df_val.rename(columns={
            'AA': 'sequence',
            'pred_halflife': 'prediction',
            'halflife': 'label'
        })[['tid', 'gene', 'sequence', 'prediction', 'label']]
        
        # Save individual fold predictions
        wt_pred_dir = os.path.join(args.output_dir, "wild_type_predictions")
        os.makedirs(wt_pred_dir, exist_ok=True)
        fold_out_path = os.path.join(wt_pred_dir, f"val_predictions_fold_{f_idx}.csv")
        df_val_filtered.to_csv(fold_out_path, index=False)
        print(f"Validierungsvorhersagen für Fold {f_idx} gespeichert unter {fold_out_path}")
        
        val_dfs.append(df_val_filtered)
        
        # Clean up
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not val_dfs:
        raise FileNotFoundError("Fehler: Für keinen der Folds konnten Validierungsvorhersagen generiert werden!")

    # Combine all validation predictions
    combined_val_df = pd.concat(val_dfs, ignore_index=True)
    wt_pred_dir = os.path.join(args.output_dir, "wild_type_predictions")
    os.makedirs(wt_pred_dir, exist_ok=True)
    combined_out_path = os.path.join(wt_pred_dir, "val_predictions_all_folds.csv")
    combined_val_df.to_csv(combined_out_path, index=False)
    print(f"\nKombinierte Out-of-Fold Validierungsvorhersagen gespeichert unter {combined_out_path}")
    
    return combined_val_df

def compile_protein_wt_predictions(args):
    """
    Loads wild type predictions and ground truths for both validation and test sets,
    mapping them back to their splits for plotting WT predictions vs Ground Truth.
    """
    print("Compiling Protein Wild-Type predictions and ground truths...")
    
    # 1. Load the splitting info
    if not os.path.exists(args.csv_path):
        print(f"Warning: split file {args.csv_path} not found. Cannot determine splits.")
        df_splits = pd.DataFrame()
    else:
        print(f"Loading split data from: {args.csv_path}")
        df_splits = pd.read_csv(args.csv_path)[['tid', 'split']].drop_duplicates()
        
    wt_pred_dir = os.path.join(args.output_dir, "wild_type_predictions")
    val_path = os.path.join(wt_pred_dir, "val_predictions_all_folds.csv")
    test_path = os.path.join(wt_pred_dir, "test_predictions_average.csv")
    
    df_val = pd.DataFrame()
    if os.path.exists(val_path):
        print(f"Loading Val WT Predictions from: {val_path}")
        df_val = pd.read_csv(val_path)
        df_val['split_source'] = 'val'
    else:
        print(f"Warning: Validation prediction file {val_path} not found.")
        
    df_test = pd.DataFrame()
    if os.path.exists(test_path):
        print(f"Loading Test WT Predictions from: {test_path}")
        df_test = pd.read_csv(test_path)
        df_test['split_source'] = 'test'
    else:
        print(f"Note: Test prediction file {test_path} not found. Skipping test set WT predictions.")
        
    # Combine predictions
    dfs_to_concat = []
    if not df_val.empty:
        dfs_to_concat.append(df_val[['tid', 'prediction', 'label']])
    if not df_test.empty:
        dfs_to_concat.append(df_test[['tid', 'prediction', 'label']])
        
    if not dfs_to_concat:
        print("Warning: No wild-type prediction files found.")
        return pd.DataFrame()
        
    df_wt = pd.concat(dfs_to_concat, ignore_index=True)
    df_wt = df_wt.drop_duplicates(subset=['tid'])
    
    # Merge splits info
    if not df_splits.empty:
        df_wt = pd.merge(df_wt, df_splits, on='tid', how='left')
    else:
        # Default split based on source if splits file not found
        df_wt['split'] = df_wt['split_source'].apply(lambda x: 8 if x == 'test' else 0)
        
    # Rename columns to match plot_wt_predictions_vs_gt.py logic
    df_wt = df_wt.rename(columns={
        'prediction': 'pred_wt',
        'label': 'label_wt',
        'split': 'data_split'
    })
    
    # Map splits to categories for plotting
    df_wt['split_type'] = df_wt['data_split'].apply(
        lambda x: 'Test Set (Split 8-9)' if x in [8, 9] else 'Validation Set (Split 0-7)'
    )
    
    # Keep only matched records
    df_wt = df_wt.dropna(subset=['pred_wt', 'label_wt']).copy()
    
    print(f"Compilation complete. Found predictions for {len(df_wt)} proteins.")
    return df_wt

def map_consequence_category(x):
    if pd.isna(x):
        return "Other"
    x = str(x).strip().lower()
    if x == 'nonsense':
        return "Truncation"
    elif x == 'missense':
        return "Exchange"
    return "Other"

def calculate_metrics(y_true, y_pred):
    """Calculates evaluation metrics."""
    pearson_r, p_val_p = pearsonr(y_true, y_pred)
    spearman_rho, p_val_s = spearmanr(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    # R^2 determination coefficient
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "N": len(y_true),
        "Pearson r": pearson_r,
        "Pearson p-value": p_val_p,
        "Spearman rho": spearman_rho,
        "Spearman p-value": p_val_s,
        "MSE": mse,
        "R2": r2
    }

def plot_wt_combined_scatter(df_wt, plots_folder, scale='log2'):
    """Generates a high-quality scatter plot for all wild types on a linear or log2 scale."""
    import matplotlib.ticker as ticker
    
    y_true = df_wt['label_wt'].values
    y_pred = df_wt['pred_wt'].values
    
    if scale == 'log2':
        # Calculate metrics in log2 space
        y_true_log = np.log2(np.clip(y_true, 1e-2, None))
        y_pred_log = np.log2(np.clip(y_pred, 1e-2, None))
        metrics = calculate_metrics(y_true_log, y_pred_log)
    else:
        # Calculate metrics in linear space
        metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n--- WT Overall Metrics (Combined, {scale} scale) ---")
    for k, v in metrics.items():
        if isinstance(v, int):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.6f}")
            
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Palette definition
    palette = {
        'Validation Set (Split 0-7)': '#2980B9',
        'Test Set (Split 8-9)': '#E67E22'
    }
    
    # Scatter plot
    sns.scatterplot(
        data=df_wt,
        x='label_wt',
        y='pred_wt',
        hue='split_type',
        palette=palette,
        alpha=0.6,
        edgecolor='none',
        s=30,
        ax=ax
    )
    
    if scale == 'log2':
        # Log scale configuration
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        
        # Line of identity (y = x) in log space
        min_val = max(min(y_true.min(), y_pred.min()), 1e-2)
        max_val = max(max(y_true.max(), y_pred.max()), 1e-2)
        log_min = np.log2(min_val)
        log_max = np.log2(max_val)
        log_padding = (log_max - log_min) * 0.05
        limits = [2 ** (log_min - log_padding), 2 ** (log_max + log_padding)]
    else:
        # Line of identity (y = x) in linear space
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        padding = (max_val - min_val) * 0.05
        limits = [min_val - padding, max_val + padding]
        
    ax.plot(limits, limits, color='#34495E', linestyle='--', linewidth=1.5, label='y = x (Identity)')
    
    # Textbox for stats
    if scale == 'log2':
        textstr = '\n'.join((
            f"N = {metrics['N']}",
            f"Pearson $r$ (log2) = {metrics['Pearson r']:.3f}",
            f"Spearman $\\rho$ = {metrics['Spearman rho']:.3f}",
            f"MSE (log2) = {metrics['MSE']:.3f}",
            f"$R^2$ (log2) = {metrics['R2']:.3f}"
        ))
    else:
        textstr = '\n'.join((
            f"N = {metrics['N']}",
            f"Pearson $r$ = {metrics['Pearson r']:.3f}",
            f"Spearman $\\rho$ = {metrics['Spearman rho']:.3f}",
            f"MSE = {metrics['MSE']:.3f}",
            f"$R^2$ = {metrics['R2']:.3f}"
        ))
        
    props = dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9', edgecolor='#BDC3C7', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    
    if scale == 'log2':
        ax.set_title('ESM Wild-Type Predictions vs. Ground Truth (All Splits - Log2 Scale)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Ground Truth Label (Actual half-life, Log2 scale)', fontsize=12)
        ax.set_ylabel('Predicted half-life (Model output, Log2 scale)', fontsize=12)
    else:
        ax.set_title('ESM Wild-Type Predictions vs. Ground Truth (All Splits - Linear Scale)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Ground Truth Label (Actual half-life)', fontsize=12)
        ax.set_ylabel('Predicted half-life (Model output)', fontsize=12)
        
    ax.legend(loc='upper right', frameon=True, facecolor='#F8F9F9', edgecolor='#BDC3C7')
    
    plt.tight_layout()
    out_path = os.path.join(plots_folder, f"wt_predictions_vs_gt_scatter_{scale}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved combined WT scatter plot to: {out_path}")
    plt.close()

def plot_wt_split_scatters(df_wt, plots_folder, scale='log2'):
    """Generates a 1x2 panel plot separating Validation and Test sets on a linear or log2 scale."""
    import matplotlib.ticker as ticker
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    
    # Subplot details
    subplots_data = [
        {
            "df": df_wt[df_wt['data_split'] < 8],
            "title": "Validation Set (Splits 0-7)",
            "color": "#2980B9",
            "ax": axes[0]
        },
        {
            "df": df_wt[df_wt['data_split'] >= 8],
            "title": "Test Set (Splits 8-9)",
            "color": "#E67E22",
            "ax": axes[1]
        }
    ]
    
    # Determine global limits for uniform comparison
    y_true_all = df_wt['label_wt'].values
    y_pred_all = df_wt['pred_wt'].values
    
    if scale == 'log2':
        min_val = max(min(y_true_all.min(), y_pred_all.min()), 1e-2)
        max_val = max(max(y_true_all.max(), y_pred_all.max()), 1e-2)
        log_min = np.log2(min_val)
        log_max = np.log2(max_val)
        log_padding = (log_max - log_min) * 0.05
        limits = [2 ** (log_min - log_padding), 2 ** (log_max + log_padding)]
    else:
        min_val = min(y_true_all.min(), y_pred_all.min())
        max_val = max(y_true_all.max(), y_pred_all.max())
        padding = (max_val - min_val) * 0.05
        limits = [min_val - padding, max_val + padding]
    
    for sub in subplots_data:
        df_sub = sub["df"]
        ax = sub["ax"]
        
        if df_sub.empty:
            ax.text(0.5, 0.5, f"No data for {sub['title']}", ha='center', va='center', fontsize=14)
            continue
            
        y_true = df_sub['label_wt'].values
        y_pred = df_sub['pred_wt'].values
        
        if scale == 'log2':
            # Calculate metrics in log2 space
            y_true_log = np.log2(np.clip(y_true, 1e-2, None))
            y_pred_log = np.log2(np.clip(y_pred, 1e-2, None))
            metrics = calculate_metrics(y_true_log, y_pred_log)
        else:
            # Calculate metrics in linear space
            metrics = calculate_metrics(y_true, y_pred)
            
        print(f"\n--- WT Metrics for {sub['title']} ({scale} scale) ---")
        for k, v in metrics.items():
            if isinstance(v, int):
                print(f"{k}: {v}")
            else:
                print(f"{k}: {v:.6f}")
                
        # Scatter
        ax.scatter(y_true, y_pred, color=sub["color"], alpha=0.5, edgecolor='none', s=25)
        
        if scale == 'log2':
            # Log scale configuration
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
            
        # Identity line
        ax.plot(limits, limits, color='#34495E', linestyle='--', linewidth=1.5, label='y = x')
        
        # Textbox
        if scale == 'log2':
            textstr = '\n'.join((
                f"N = {metrics['N']}",
                f"Pearson $r$ (log2) = {metrics['Pearson r']:.3f}",
                f"Spearman $\\rho$ = {metrics['Spearman rho']:.3f}",
                f"MSE (log2) = {metrics['MSE']:.3f}",
                f"$R^2$ (log2) = {metrics['R2']:.3f}"
            ))
        else:
            textstr = '\n'.join((
                f"N = {metrics['N']}",
                f"Pearson $r$ = {metrics['Pearson r']:.3f}",
                f"Spearman $\\rho$ = {metrics['Spearman rho']:.3f}",
                f"MSE = {metrics['MSE']:.3f}",
                f"$R^2$ = {metrics['R2']:.3f}"
            ))
            
        props = dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9', edgecolor='#BDC3C7', alpha=0.9)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_title(sub["title"], fontsize=13, fontweight='bold')
        if scale == 'log2':
            ax.set_xlabel('Ground Truth Label (Log2 scale)', fontsize=11)
            ax.set_ylabel('Predicted Value (Log2 scale)', fontsize=11)
        else:
            ax.set_xlabel('Ground Truth Label', fontsize=11)
            ax.set_ylabel('Predicted Value', fontsize=11)
        ax.legend(loc='upper right', frameon=True)
        
    title_suffix = ' (Log2 Scale)' if scale == 'log2' else ' (Linear Scale)'
    plt.suptitle(f'ESM Predictions vs. Ground Truth by Data Split{title_suffix}', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(plots_folder, f"wt_predictions_vs_gt_subplots_{scale}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved side-by-side WT scatter plots to: {out_path}")
    plt.close()

def plot_variant_significance(df_plot, plots_folder, scale='linear'):
    """Generates the significance-based variant plots (Scatter and Delta distribution)."""
    print(f"\nStarting Variant Significance Visualizations ({scale} scale)...")
    
    # Log counts
    print(f"Number of variants included: {len(df_plot)}")
    print(df_plot['category'].value_counts())
    
    palette = {"Pathogenic": "red", "Benign": "green"}
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: Main Scatter Plot (Multi-layout) ---
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    
    # 1.1 All Variants (Large Top)
    ax_top = fig.add_subplot(gs[0, :])
    
    # Prepare data for plotting (with optional clipping for log scale)
    if scale == 'log2':
        plot_data = df_plot.copy()
        plot_data['pred_wt'] = np.clip(plot_data['pred_wt'], 1e-2, None)
        plot_data['pred_mut_mean'] = np.clip(plot_data['pred_mut_mean'], 1e-2, None)
    else:
        plot_data = df_plot
        
    sns.scatterplot(data=plot_data, x='pred_wt', y='pred_mut_mean', hue='category', palette=palette, alpha=0.6, ax=ax_top)
    
    # helper for y=x line
    def add_yx_line(ax, data_x, data_y):
        if scale == 'log2':
            min_v = max(min(data_x.min(), data_y.min()), 1e-2)
            max_v = max(max(data_x.max(), data_y.max()), 1e-2)
        else:
            min_v = min(data_x.min(), data_y.min())
            max_v = max(data_x.max(), data_y.max())
        ax.plot([min_v, max_v], [min_v, max_v], color='black', linestyle='--', label='y=x')

    add_yx_line(ax_top, plot_data['pred_wt'], plot_data['pred_mut_mean'])
    
    title_suffix = ' (Log2 Scale)' if scale == 'log2' else ' (Linear Scale)'
    ax_top.set_title(f'All Variants: Wild-type vs. Mutated Predictions{title_suffix} (n={len(df_plot)})', fontsize=16)
    
    if scale == 'log2':
        import matplotlib.ticker as ticker
        ax_top.set_xscale('log', base=2)
        ax_top.set_yscale('log', base=2)
        ax_top.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_top.set_xlabel('Wild-type Prediction (Log2 scale)')
        ax_top.set_ylabel('Mutated Prediction (Mean, Log2 scale)')
    else:
        ax_top.set_xlabel('Wild-type Prediction')
        ax_top.set_ylabel('Mutated Prediction (Mean)')
    ax_top.legend()

    # 1.2 Benign Only (Small Bottom Left)
    ax_benign = fig.add_subplot(gs[1, 0])
    df_benign = plot_data[plot_data['category'] == "Benign"]
    if not df_benign.empty:
        sns.scatterplot(data=df_benign, x='pred_wt', y='pred_mut_mean', color='green', alpha=0.5, ax=ax_benign)
        add_yx_line(ax_benign, df_benign['pred_wt'], df_benign['pred_mut_mean'])
        
    ax_benign.set_title(f'Benign Variants Only (n={len(df_benign)})', fontsize=12)
    if scale == 'log2':
        import matplotlib.ticker as ticker
        ax_benign.set_xscale('log', base=2)
        ax_benign.set_yscale('log', base=2)
        ax_benign.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_benign.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_benign.set_xlabel('WT Prediction (Log2 scale)')
        ax_benign.set_ylabel('Mutated Prediction (Log2 scale)')
    else:
        ax_benign.set_xlabel('WT Prediction')
        ax_benign.set_ylabel('Mutated Prediction')

    # 1.3 Pathogenic Only (Small Bottom Right)
    ax_patho = fig.add_subplot(gs[1, 1])
    df_patho = plot_data[plot_data['category'] == "Pathogenic"]
    if not df_patho.empty:
        sns.scatterplot(data=df_patho, x='pred_wt', y='pred_mut_mean', color='red', alpha=0.5, ax=ax_patho)
        add_yx_line(ax_patho, df_patho['pred_wt'], df_patho['pred_mut_mean'])
        
    ax_patho.set_title(f'Pathogenic Variants Only (n={len(df_patho)})', fontsize=12)
    if scale == 'log2':
        import matplotlib.ticker as ticker
        ax_patho.set_xscale('log', base=2)
        ax_patho.set_yscale('log', base=2)
        ax_patho.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_patho.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_patho.set_xlabel('WT Prediction (Log2 scale)')
        ax_patho.set_ylabel('Mutated Prediction (Log2 scale)')
    else:
        ax_patho.set_xlabel('WT Prediction')
        ax_patho.set_ylabel('Mutated Prediction')

    plt.tight_layout()
    out1 = os.path.join(plots_folder, f"scatter_wt_vs_mut_main_{scale}.png")
    plt.savefig(out1, dpi=300)
    print(f"Saved: {out1}")
    plt.close()

    # --- PLOT 2: Delta Distribution ---
    if scale == 'linear':
        plt.figure(figsize=(10, 6))
        df_plot_kde = df_plot.dropna(subset=['delta', 'category']).copy()
        total_n = len(df_plot_kde)
        category_counts = df_plot_kde['category'].value_counts()
        
        kde_palette = {}
        kde_hue_col = []
        for cat in df_plot_kde['category']:
            count = category_counts.get(cat, 0)
            label = f"{cat} (n={count})"
            kde_hue_col.append(label)
            if cat in palette:
                kde_palette[label] = palette[cat]
        df_plot_kde['category_legend'] = kde_hue_col

        sns.kdeplot(data=df_plot_kde, x='delta', hue='category_legend', palette=kde_palette, fill=True, common_norm=False)
        plt.axvline(0, color='gray', linestyle='--')
        plt.title(f'Distribution of Prediction Changes (Delta) (n={total_n})', fontsize=15)
        plt.xlabel('Delta (Mutated - Wild-type)', fontsize=12)
        
        out2 = os.path.join(plots_folder, "delta_distribution_significance.png")
        plt.savefig(out2, dpi=300)
        print(f"Saved: {out2}")
        plt.close()

def plot_variant_consequence(df_plot, plots_folder, scale='linear'):
    """Generates the consequence-based variant plots (Scatter and Delta distribution)."""
    print(f"\nStarting Consequence Visualizations (Truncation vs Exchange) ({scale} scale)...")
    
    # Filter for Truncation and Exchange only
    df_conseq = df_plot[df_plot['consequence_category'].isin(["Truncation", "Exchange"])].copy()
    
    # Log counts
    conseq_counts = df_conseq['consequence_category'].value_counts()
    print("Consequence counts:")
    print(conseq_counts)
    
    if df_conseq.empty:
        print("Warning: No Exchange or Truncation consequence categories found. Skipping consequence plots.")
        return
        
    # Color palette
    palette = {"Truncation": "#d62728", "Exchange": "#1f77b4"}
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: Main Scatter Plot (Truncation vs Exchange) ---
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    
    # 1.1 All Variants (Large Top)
    ax_top = fig.add_subplot(gs[0, :])
    
    # Prepare data for plotting (with optional clipping for log scale)
    if scale == 'log2':
        plot_data = df_conseq.copy()
        plot_data['pred_wt'] = np.clip(plot_data['pred_wt'], 1e-2, None)
        plot_data['pred_mut_mean'] = np.clip(plot_data['pred_mut_mean'], 1e-2, None)
    else:
        plot_data = df_conseq
        
    sns.scatterplot(data=plot_data, x='pred_wt', y='pred_mut_mean', hue='consequence_category', palette=palette, alpha=0.6, ax=ax_top)
    
    # helper for y=x line
    def add_yx_line(ax, data_x, data_y):
        if scale == 'log2':
            min_v = max(min(data_x.min(), data_y.min()), 1e-2)
            max_v = max(max(data_x.max(), data_y.max()), 1e-2)
        else:
            min_v = min(data_x.min(), data_y.min())
            max_v = max(data_x.max(), data_y.max())
        ax.plot([min_v, max_v], [min_v, max_v], color='black', linestyle='--', label='y=x')

    add_yx_line(ax_top, plot_data['pred_wt'], plot_data['pred_mut_mean'])
    
    title_suffix = ' (Log2 Scale)' if scale == 'log2' else ' (Linear Scale)'
    ax_top.set_title(f'All Variants: Wild-type vs. Mutated Predictions by Consequence{title_suffix} (n={len(df_conseq)})', fontsize=16)
    
    if scale == 'log2':
        import matplotlib.ticker as ticker
        ax_top.set_xscale('log', base=2)
        ax_top.set_yscale('log', base=2)
        ax_top.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_top.set_xlabel('Wild-type Prediction (Log2 scale)')
        ax_top.set_ylabel('Mutated Prediction (Mean, Log2 scale)')
    else:
        ax_top.set_xlabel('Wild-type Prediction')
        ax_top.set_ylabel('Mutated Prediction (Mean)')
    ax_top.legend()

    # 1.2 Exchange Only (Small Bottom Left)
    ax_exchange = fig.add_subplot(gs[1, 0])
    df_exchange = plot_data[plot_data['consequence_category'] == "Exchange"]
    if not df_exchange.empty:
        sns.scatterplot(data=df_exchange, x='pred_wt', y='pred_mut_mean', color=palette["Exchange"], alpha=0.5, ax=ax_exchange)
        add_yx_line(ax_exchange, df_exchange['pred_wt'], df_exchange['pred_mut_mean'])
        
    ax_exchange.set_title(f'Exchange Variants Only (n={len(df_exchange)})', fontsize=12)
    if scale == 'log2':
        import matplotlib.ticker as ticker
        ax_exchange.set_xscale('log', base=2)
        ax_exchange.set_yscale('log', base=2)
        ax_exchange.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_exchange.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_exchange.set_xlabel('WT Prediction (Log2 scale)')
        ax_exchange.set_ylabel('Mutated Prediction (Log2 scale)')
    else:
        ax_exchange.set_xlabel('WT Prediction')
        ax_exchange.set_ylabel('Mutated Prediction')

    # 1.3 Truncation Only (Small Bottom Right)
    ax_trunc = fig.add_subplot(gs[1, 1])
    df_trunc = plot_data[plot_data['consequence_category'] == "Truncation"]
    if not df_trunc.empty:
        sns.scatterplot(data=df_trunc, x='pred_wt', y='pred_mut_mean', color=palette["Truncation"], alpha=0.5, ax=ax_trunc)
        add_yx_line(ax_trunc, df_trunc['pred_wt'], df_trunc['pred_mut_mean'])
        
    ax_trunc.set_title(f'Truncation Variants Only (n={len(df_trunc)})', fontsize=12)
    if scale == 'log2':
        import matplotlib.ticker as ticker
        ax_trunc.set_xscale('log', base=2)
        ax_trunc.set_yscale('log', base=2)
        ax_trunc.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_trunc.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:g}"))
        ax_trunc.set_xlabel('WT Prediction (Log2 scale)')
        ax_trunc.set_ylabel('Mutated Prediction (Log2 scale)')
    else:
        ax_trunc.set_xlabel('WT Prediction')
        ax_trunc.set_ylabel('Mutated Prediction')

    plt.tight_layout()
    out1 = os.path.join(plots_folder, f"scatter_wt_vs_mut_consequence_main_{scale}.png")
    plt.savefig(out1, dpi=300)
    print(f"Saved: {out1}")
    plt.close()

    # --- PLOT 2: Delta Distribution ---
    if scale == 'linear':
        plt.figure(figsize=(10, 6))
        df_conseq_kde = df_conseq.dropna(subset=['delta', 'consequence_category']).copy()
        total_n = len(df_conseq_kde)
        conseq_counts = df_conseq_kde['consequence_category'].value_counts()
        
        kde_palette = {}
        kde_hue_col = []
        for cat in df_conseq_kde['consequence_category']:
            count = conseq_counts.get(cat, 0)
            label = f"{cat} (n={count})"
            kde_hue_col.append(label)
            if cat in palette:
                kde_palette[label] = palette[cat]
        df_conseq_kde['consequence_legend'] = kde_hue_col

        sns.kdeplot(data=df_conseq_kde, x='delta', hue='consequence_legend', palette=kde_palette, fill=True, common_norm=False)
        plt.axvline(0, color='gray', linestyle='--')
        plt.title(f'Distribution of Prediction Changes (Delta) by Consequence Class (n={total_n})', fontsize=15)
        plt.xlabel('Delta (Mutated - Wild-type)', fontsize=12)
        
        out2 = os.path.join(plots_folder, "delta_distribution_consequence.png")
        plt.savefig(out2, dpi=300)
        print(f"Saved: {out2}")
        plt.close()

def run_analysis(args):
    print("\n==================== Starte Varianten-Vorhersage-Analyse ====================")
    
    combined_val_path = os.path.join(args.output_dir, "wild_type_predictions", "val_predictions_all_folds.csv")
    if not os.path.exists(combined_val_path):
        raise FileNotFoundError(f"Fehler: {combined_val_path} existiert nicht. Bitte lassen Sie zuerst die Validierungsvorhersagen laufen.")
        
    if not os.path.exists(args.mutated_csv_path):
        raise FileNotFoundError(f"Fehler: {args.mutated_csv_path} existiert nicht.")
        
    val_df = pd.read_csv(combined_val_path)
    mut_df = pd.read_csv(args.mutated_csv_path)
    
    print(f"Lade Wild-Type Vorhersagen: {len(val_df)} Zeilen")
    print(f"Lade Mutierte Vorhersagen: {len(mut_df)} Zeilen")
    
    # Merge on tid
    # Handle possible differences in WT prediction column names ('prediction'/'pred_halflife' and 'label'/'halflife')
    val_wt_cols = ['tid', 'gene']
    if 'prediction' in val_df.columns:
        val_wt_cols.append('prediction')
    elif 'pred_halflife' in val_df.columns:
        val_df = val_df.rename(columns={'pred_halflife': 'prediction'})
        val_wt_cols.append('prediction')
    else:
        raise KeyError("Fehler: Weder 'prediction' noch 'pred_halflife' in Wild-Type Vorhersagen gefunden.")
        
    if 'label' in val_df.columns:
        val_wt_cols.append('label')
    elif 'halflife' in val_df.columns:
        val_df = val_df.rename(columns={'halflife': 'label'})
        val_wt_cols.append('label')
    else:
        raise KeyError("Fehler: Weder 'label' noch 'halflife' in Wild-Type Vorhersagen gefunden.")

    df_merged = pd.merge(
        val_df[val_wt_cols],
        mut_df[['tid', 'clinvar_id', 'clinical_significance', 'pred_mut_halflife']],
        on='tid'
    )
    
    print(f"Anzahl erfolgreich gematchter Varianten: {len(df_merged)}")
    if len(df_merged) == 0:
        print("Warnung: Keine Übereinstimmungen auf 'tid' zwischen Wild-Type und mutierten Sequenzen gefunden.")
        return
        
    # Calculate delta
    df_merged['delta_halflife'] = df_merged['pred_mut_halflife'] - df_merged['prediction']
    
    # Save the dataframe with all predicted half-lives of WT and variants
    variant_predictions_dir = os.path.join(args.output_dir, "variant_predictions")
    os.makedirs(variant_predictions_dir, exist_ok=True)
    processed_results_path = os.path.join(variant_predictions_dir, "processed_mutation_results.csv")
    
    df_save = df_merged.copy()
    # Add alias columns to align with the RNA results structure if needed
    df_save['pred_wt'] = df_save['prediction']
    df_save['pred_mut_mean'] = df_save['pred_mut_halflife']
    
    df_save.to_csv(processed_results_path, index=False)
    print(f"Gemergeder DataFrame mit Vorhersagen unter {processed_results_path} gespeichert.")
    
    benign = df_merged[df_merged['clinical_significance'] == 'Benign']
    pathogenic = df_merged[df_merged['clinical_significance'] == 'Pathogenic']

    # Generate Plots
    try:
        import matplotlib
        matplotlib.use('Agg') # Force non-interactive backend for server compatibility
        import matplotlib.pyplot as plt
        
        plots_dir = os.path.join(args.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Boxplot of Delta Halflife
        plt.figure(figsize=(8, 6))
        data_to_plot = []
        labels = []
        if len(benign) > 0:
            data_to_plot.append(benign['delta_halflife'].dropna())
            labels.append(f'Benign (n={len(benign)})')
        if len(pathogenic) > 0:
            data_to_plot.append(pathogenic['delta_halflife'].dropna())
            labels.append(f'Pathogenic (n={len(pathogenic)})')
            
        plt.boxplot(data_to_plot, labels=labels)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.ylabel('Delta Predicted Half-life (Mutated - WT)')
        plt.title(f'Impact of Mutations on Predicted Protein Half-life (N={len(df_merged)})')
        plt.grid(True, alpha=0.3)
        
        boxplot_path = os.path.join(plots_dir, "variant_prediction_boxplot.png")
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Boxplot gespeichert unter {boxplot_path}")
        
        # 1b. Violin plot of Delta Halflife
        plt.figure(figsize=(8, 6))
        data_to_plot_val = []
        if len(benign) > 0:
            data_to_plot_val.append(benign['delta_halflife'].dropna().values)
        if len(pathogenic) > 0:
            data_to_plot_val.append(pathogenic['delta_halflife'].dropna().values)
            
        if data_to_plot_val:
            parts = plt.violinplot(data_to_plot_val, showmeans=False, showmedians=True, showextrema=True)
            
            colors = []
            if len(benign) > 0:
                colors.append('green')
            if len(pathogenic) > 0:
                colors.append('red')
                
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_edgecolor('black')
                pc.set_alpha(0.5)
                
            for key in ['cmaxes', 'cmins', 'cbars', 'cmedians']:
                if key in parts:
                    parts[key].set_edgecolor('black')
                    parts[key].set_linewidth(1.0)
                    
            plt.xticks(range(1, len(labels) + 1), labels)
            
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.ylabel('Delta Predicted Half-life (Mutated - WT)')
        plt.title(f'Impact of Mutations on Predicted Protein Half-life (N={len(df_merged)})')
        plt.grid(True, alpha=0.3)
        
        violin_path = os.path.join(plots_dir, "variant_prediction_violin.png")
        plt.savefig(violin_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Violinplot gespeichert unter {violin_path}")
        
        # Note: Pre-existing duplicate scatter plots (variant_prediction_scatter.png and 
        # wild_type_gt_vs_pred_scatter.png) have been removed in favor of the new, standardized 
        # plotting functions below.

        # --- New WT vs GT Plots (Validation & Test combined and split) ---
        try:
            df_wt = compile_protein_wt_predictions(args)
            if not df_wt.empty:
                for scale in ['linear', 'log2']:
                    plot_wt_combined_scatter(df_wt, plots_dir, scale=scale)
                    plot_wt_split_scatters(df_wt, plots_dir, scale=scale)
            else:
                print("No matching WT predictions were compiled for combined WT plots.")
        except Exception as wt_err:
            print(f"Warnung: Die neuen WT-Plots konnten nicht erzeugt werden: {wt_err}")
            import traceback
            traceback.print_exc()

        # --- New Variant Plots (Significance & Consequence) ---
        try:
            # Prepare data for new variant plots
            df_plot = df_merged.copy()
            df_plot['pred_wt'] = df_plot['prediction']
            df_plot['pred_mut_mean'] = df_plot['pred_mut_halflife']
            df_plot['category'] = df_plot['clinical_significance']
            df_plot['delta'] = df_plot['pred_mut_mean'] - df_plot['pred_wt']
            
            # Merge mutated metadata to get consequence mapping
            mutated_csv_meta = "/beegfs/prj/RNA_NLP/protein_half_lives/esm_data/Protein_half_lifes_mutated.csv"
            if mutated_csv_meta and os.path.exists(mutated_csv_meta):
                print(f"Loading mutated metadata from: {mutated_csv_meta} to map mutation consequences...")
                df_mut_meta = pd.read_csv(mutated_csv_meta)[['tid', 'clinvar_id', 'mutation_type']].drop_duplicates()
                df_plot = pd.merge(df_plot, df_mut_meta, on=['tid', 'clinvar_id'], how='left')
                df_plot['consequence_category'] = df_plot['mutation_type'].apply(map_consequence_category)
            else:
                print("Warning: Mutated metadata file (Protein_half_lifes_mutated.csv) not found. Consequence plots will be skipped.")
                df_plot['consequence_category'] = "Other"
                
            # Plot significance and consequence plots for both scales
            for scale in ['linear', 'log2']:
                plot_variant_significance(df_plot, plots_dir, scale=scale)
                
                # Plot consequence plots if we have the consequence_category column loaded
                if 'consequence_category' in df_plot.columns and df_plot['consequence_category'].nunique() > 1:
                    plot_variant_consequence(df_plot, plots_dir, scale=scale)
        except Exception as var_err:
            print(f"Warnung: Die neuen Varianten-Plots (Significance/Consequence) konnten nicht erzeugt werden: {var_err}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Warnung: Visualisierungen konnten nicht erzeugt werden. Fehler: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze wild-type vs mutated sequence predictions.")
    parser.add_argument("--csv_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/Protein_half_lifes.csv", help="Pfad zur Protein_half_lifes.csv")
    parser.add_argument("--mutated_csv_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_output/variant_predictions/variants_prediction_average.csv", help="Pfad zur variants_prediction_average.csv")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t12_35M_UR50D", help="ESM Modellname von Hugging Face")
    parser.add_argument("--cache_dir", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_weights", help="Speicherort für Hugging Face Gewichte")
    parser.add_argument("--output_dir", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_output", help="Ausgabeverzeichnis für die Ergebnisse")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Größe für Inferenz")
    parser.add_argument("--create_val_predictions", action="store_true", help="Generiere die Validierungsvorhersagen für normale Sequenzen.")
    
    args = parser.parse_args()

    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    os.environ['HF_HOME'] = args.cache_dir
    os.makedirs(args.output_dir, exist_ok=True)

    folds = {
        0: {'train': [0, 1, 2, 3, 4, 5], 'val': [6, 7]},
        1: {'train': [0, 1, 2, 3, 6, 7], 'val': [4, 5]},
        2: {'train': [0, 1, 4, 5, 6, 7], 'val': [2, 3]},
        3: {'train': [2, 3, 4, 5, 6, 7], 'val': [0, 1]}
    }

    # 1. Optionale Generierung der WT Validierungsvorhersagen
    if args.create_val_predictions:
        print("Starte Generierung der Validierungsvorhersagen für normale Sequenzen...")
        print(f"Lade Tokenizer: {args.model_name}")
        tokenizer = EsmTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        generate_validation_predictions(args, tokenizer, folds)
    else:
        print("Überspringe Generierung der Validierungsvorhersagen.")

    # 2. Führe Analyse durch
    run_analysis(args)

if __name__ == "__main__":
    main()
