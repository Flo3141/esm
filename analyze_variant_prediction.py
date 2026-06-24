import os
import argparse
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmForSequenceClassification
import gc

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
        
        # Save individual fold predictions
        fold_out_path = os.path.join(args.output_dir, f"val_predictions_fold_{f_idx}.csv")
        df_val.to_csv(fold_out_path, index=False)
        print(f"Validierungsvorhersagen für Fold {f_idx} gespeichert unter {fold_out_path}")
        
        val_dfs.append(df_val)
        
        # Clean up
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not val_dfs:
        raise FileNotFoundError("Fehler: Für keinen der Folds konnten Validierungsvorhersagen generiert werden!")

    # Combine all validation predictions
    combined_val_df = pd.concat(val_dfs, ignore_index=True)
    combined_out_path = os.path.join(args.output_dir, "val_predictions_all_folds.csv")
    combined_val_df.to_csv(combined_out_path, index=False)
    print(f"\nKombinierte Out-of-Fold Validierungsvorhersagen gespeichert unter {combined_out_path}")
    
    return combined_val_df

def average_test_predictions(output_dir, folds_keys):
    print("\n==================== Berechne durchschnittliche Test-Vorhersagen ====================")
    dfs = []
    for f_idx in folds_keys:
        path = os.path.join(output_dir, f"test_predictions_fold_{f_idx}.csv")
        if not os.path.exists(path):
            print(f"Warnung: Test-Vorhersagedatei nicht gefunden: {path}")
            continue
        df = pd.read_csv(path)
        # Rename prediction to prediction_fold_x to prevent conflict
        df = df.rename(columns={"prediction": f"prediction_fold_{f_idx}"})
        dfs.append(df)
    
    if not dfs:
        print("Keine Test-Vorhersagedateien zum Mitteln gefunden.")
        return
        
    # Merge all DataFrames on keys to align rows exactly
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['tid', 'gene', 'sequence', 'label'])
        
    # Calculate average prediction
    pred_cols = [f"prediction_fold_{f_idx}" for f_idx in folds_keys if f"prediction_fold_{f_idx}" in merged_df.columns]
    merged_df['prediction'] = merged_df[pred_cols].mean(axis=1)
    
    # Keep only target columns
    final_df = merged_df[['tid', 'gene', 'sequence', 'prediction', 'label']]
    
    # Save to output file
    out_path = os.path.join(output_dir, "test_predictions_average.csv")
    final_df.to_csv(out_path, index=False)
    print(f"Durchschnittliche Test-Vorhersagen gespeichert unter {out_path}")

def run_analysis(args):
    print("\n==================== Starte Varianten-Vorhersage-Analyse ====================")
    
    combined_val_path = os.path.join(args.output_dir, "val_predictions_all_folds.csv")
    if not os.path.exists(combined_val_path):
        raise FileNotFoundError(f"Fehler: {combined_val_path} existiert nicht. Bitte lassen Sie zuerst die Validierungsvorhersagen laufen.")
        
    if not os.path.exists(args.mutated_csv_path):
        raise FileNotFoundError(f"Fehler: {args.mutated_csv_path} existiert nicht.")
        
    val_df = pd.read_csv(combined_val_path)
    mut_df = pd.read_csv(args.mutated_csv_path)
    
    print(f"Lade Wild-Type Vorhersagen: {len(val_df)} Zeilen")
    print(f"Lade Mutierte Vorhersagen: {len(mut_df)} Zeilen")
    
    # Merge on tid
    df_merged = pd.merge(
        val_df[['tid', 'gene', 'halflife', 'pred_halflife']],
        mut_df[['tid', 'variant_id', 'clinical_significance', 'pred_mut_halflife']],
        on='tid'
    )
    
    print(f"Anzahl erfolgreich gematchter Varianten: {len(df_merged)}")
    if len(df_merged) == 0:
        print("Warnung: Keine Übereinstimmungen auf 'tid' zwischen Wild-Type und mutierten Sequenzen gefunden.")
        return
        
    # Calculate delta
    df_merged['delta_halflife'] = df_merged['pred_mut_halflife'] - df_merged['pred_halflife']
    
    benign = df_merged[df_merged['clinical_significance'] == 'Benign']
    pathogenic = df_merged[df_merged['clinical_significance'] == 'Pathogenic']
    
    # Statistical significance test
    test_str = ""
    try:
        from scipy.stats import mannwhitneyu
        if len(pathogenic) > 0 and len(benign) > 0:
            stat, p_val = mannwhitneyu(pathogenic['delta_halflife'], benign['delta_halflife'], alternative='two-sided')
            test_str = f"Mann-Whitney U Test: U-statistic = {stat:.2f}, p-value = {p_val:.2e}\n"
        else:
            test_str = "Mann-Whitney U Test: Kann nicht durchgeführt werden, da eine der Klassen leer ist.\n"
    except ImportError:
        test_str = "Mann-Whitney U Test: scipy.stats.mannwhitneyu konnte nicht importiert werden.\n"

    # Print & Save Summary
    summary_text = "="*65 + "\n"
    summary_text += "ANALYSIS OF VARIANT PREDICTIONS (WILD-TYPE VS MUTATED)\n"
    summary_text += "="*65 + "\n\n"
    summary_text += f"Gesamtanzahl gematchter Varianten: {len(df_merged)}\n"
    summary_text += f"Pathogene Varianten: {len(pathogenic)}\n"
    summary_text += f"Benigne Varianten: {len(benign)}\n\n"
    
    summary_text += "--- Delta Halbwertszeit (Mutiert - Wildtyp) Summary Statistics ---\n"
    summary_text += f"\nPathogene Varianten:\n{pathogenic['delta_halflife'].describe().to_string()}\n"
    summary_text += f"\nBenigne Varianten:\n{benign['delta_halflife'].describe().to_string()}\n\n"
    
    summary_text += "--- Statistische Signifikanz ---\n"
    summary_text += test_str
    
    print(summary_text)
    
    summary_path = os.path.join(args.output_dir, "variant_analysis_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"Analyse-Zusammenfassung gespeichert unter {summary_path}")

    # Generate Plots
    try:
        import matplotlib
        matplotlib.use('Agg') # Force non-interactive backend for server compatibility
        import matplotlib.pyplot as plt
        
        # 1. Boxplot of Delta Halflife
        plt.figure(figsize=(8, 6))
        data_to_plot = []
        labels = []
        if len(benign) > 0:
            data_to_plot.append(benign['delta_halflife'].dropna())
            labels.append('Benign')
        if len(pathogenic) > 0:
            data_to_plot.append(pathogenic['delta_halflife'].dropna())
            labels.append('Pathogenic')
            
        plt.boxplot(data_to_plot, labels=labels)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plt.ylabel('Delta Predicted Half-life (Mutated - WT)')
        plt.title('Impact of Mutations on Predicted Protein Half-life')
        plt.grid(True, alpha=0.3)
        
        boxplot_path = os.path.join(args.output_dir, "variant_prediction_analysis.png")
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
        plt.title('Impact of Mutations on Predicted Protein Half-life')
        plt.grid(True, alpha=0.3)
        
        violin_path = os.path.join(args.output_dir, "variant_prediction_violin.png")
        plt.savefig(violin_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Violinplot gespeichert unter {violin_path}")
        
        # 2. Scatter plot (WT vs Mutated) - Multi-panel layout
        plt.figure(figsize=(12, 12))
        
        # Draw identity line
        min_val = min(df_merged['pred_halflife'].min(), df_merged['pred_mut_halflife'].min())
        max_val = max(df_merged['pred_halflife'].max(), df_merged['pred_mut_halflife'].max())
        
        # --- Top Plot (Combined) ---
        ax1 = plt.subplot(2, 2, (1, 2))
        if len(benign) > 0:
            ax1.scatter(benign['pred_halflife'], benign['pred_mut_halflife'], color='green', alpha=0.5, label='Benign', s=15)
        if len(pathogenic) > 0:
            ax1.scatter(pathogenic['pred_halflife'], pathogenic['pred_mut_halflife'], color='red', alpha=0.5, label='Pathogenic', s=15)
        ax1.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='y = x (no change)', linewidth=1.2)
        ax1.set_xlabel('Wild-Type (WT) Prediction')
        ax1.set_ylabel('Mutation Prediction')
        ax1.set_title('WT vs. Mutated Half-life Prediction (All Variants)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # --- Bottom Left Plot (Benign only) ---
        ax2 = plt.subplot(2, 2, 3)
        if len(benign) > 0:
            ax2.scatter(benign['pred_halflife'], benign['pred_mut_halflife'], color='green', alpha=0.5, label='Benign', s=15)
        ax2.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='y = x (no change)', linewidth=1.2)
        ax2.set_xlabel('Wild-Type (WT) Prediction')
        ax2.set_ylabel('Mutation Prediction')
        ax2.set_title('Benign Variants')
        ax2.grid(True, alpha=0.3)
        
        # --- Bottom Right Plot (Pathogenic only) ---
        ax3 = plt.subplot(2, 2, 4)
        if len(pathogenic) > 0:
            ax3.scatter(pathogenic['pred_halflife'], pathogenic['pred_mut_halflife'], color='red', alpha=0.5, label='Pathogenic', s=15)
        ax3.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='y = x (no change)', linewidth=1.2)
        ax3.set_xlabel('Wild-Type (WT) Prediction')
        ax3.set_ylabel('Mutation Prediction')
        ax3.set_title('Pathogenic Variants')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        scatterplot_path = os.path.join(args.output_dir, "variant_prediction_scatter.png")
        plt.savefig(scatterplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatterplot gespeichert unter {scatterplot_path}")

    except Exception as e:
        print(f"Warnung: Visualisierungen konnten nicht erzeugt werden. Fehler: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze wild-type vs mutated sequence predictions.")
    parser.add_argument("--csv_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/Protein_half_lifes.csv", help="Pfad zur Protein_half_lifes.csv")
    parser.add_argument("--mutated_csv_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_output/Protein_half_lifes_predicted.csv", help="Pfad zur Protein_half_lifes_predicted.csv")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t12_35M_UR50D", help="ESM Modellname von Hugging Face")
    parser.add_argument("--cache_dir", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_weights", help="Speicherort für Hugging Face Gewichte")
    parser.add_argument("--output_dir", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_output", help="Ausgabeverzeichnis für die Ergebnisse")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Größe für Inferenz")
    parser.add_argument("--skip_prediction", action="store_true", help="Überspringe die Generierung der Validierungsvorhersagen, falls bereits erzeugt.")
    
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
    if not args.skip_prediction:
        print("Starte Generierung der Validierungsvorhersagen für normale Sequenzen...")
        print(f"Lade Tokenizer: {args.model_name}")
        tokenizer = EsmTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        generate_validation_predictions(args, tokenizer, folds)
    else:
        print("Überspringe Generierung der Validierungsvorhersagen.")

    # 2. Berechne durchschnittliche Test-Vorhersagen
    average_test_predictions(args.output_dir, list(folds.keys()))

    # 3. Führe Analyse durch
    run_analysis(args)

if __name__ == "__main__":
    main()
