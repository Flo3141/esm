#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

class DualLogger:
    """Redirects stdout to both console and a log file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# Set matplotlib backend for cluster/headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_correlations(df, col_x, col_y, label="Dataset"):
    """Helper to calculate and print Pearson and Spearman correlations."""
    df_clean = df.dropna(subset=[col_x, col_y])
    n = len(df_clean)
    if n < 3:
        print(f"[{label}] Too few data points ({n}) to compute correlation.")
        return None
        
    p_r, p_val_p = pearsonr(df_clean[col_x], df_clean[col_y])
    s_r, p_val_s = spearmanr(df_clean[col_x], df_clean[col_y])
    
    print(f"[{label}] N = {n}")
    print(f"  Pearson r  = {p_r:+.4f} (p-value: {p_val_p:.2e})")
    print(f"  Spearman r = {s_r:+.4f} (p-value: {p_val_s:.2e})")
    return {"N": n, "pearson_r": p_r, "pearson_p": p_val_p, "spearman_r": s_r, "spearman_p": p_val_s}

def analyze_hypothesis(df):
    """
    Analyzes the hypothesis: Does a decrease in RNA half-life (delta_rna < 0)
    lead to an increase in protein half-life (delta_protein > 0)?
    """
    print("\n" + "="*60)
    print("HYPOTHESIS ANALYSIS: If RNA half-life sinks, does Protein half-life rise?")
    print("="*60)
    
    # Separate into groups
    dec_rna = df[df['delta_rna'] < 0]
    inc_rna = df[df['delta_rna'] > 0]
    unc_rna = df[df['delta_rna'] == 0]
    
    print(f"Total variants with delta data: {len(df)}")
    print(f"  - RNA half-life decreases (delta_rna < 0): n = {len(dec_rna)}")
    print(f"  - RNA half-life increases (delta_rna > 0): n = {len(inc_rna)}")
    print(f"  - RNA half-life unchanged (delta_rna = 0): n = {len(unc_rna)}")
    
    for group_name, group_df in [("RNA Decreased", dec_rna), ("RNA Increased", inc_rna)]:
        if len(group_df) == 0:
            continue
        
        # Protein change stats
        delta_prot = group_df['delta_protein'].dropna()
        n_prot = len(delta_prot)
        if n_prot == 0:
            continue
            
        mean_dp = delta_prot.mean()
        median_dp = delta_prot.median()
        std_dp = delta_prot.std()
        
        n_rise = (delta_prot > 0).sum()
        n_sink = (delta_prot < 0).sum()
        n_flat = (delta_prot == 0).sum()
        
        print(f"\nStats of Protein Delta for group '{group_name}' (n = {n_prot}):")
        print(f"  Protein Delta: Mean = {mean_dp:+.4f}, Median = {median_dp:+.4f}, Std = {std_dp:.4f}")
        print(f"  Protein half-life rises   (delta > 0): {n_rise} ({n_rise/n_prot*100:.1f}%)")
        print(f"  Protein half-life sinks   (delta < 0): {n_sink} ({n_sink/n_prot*100:.1f}%)")
        print(f"  Protein half-life flat    (delta = 0): {n_flat} ({n_flat/n_prot*100:.1f}%)")

    # Statistical Significance: Mann-Whitney U test between Decreased and Increased RNA groups
    if len(dec_rna) >= 5 and len(inc_rna) >= 5:
        u_stat, p_val = mannwhitneyu(dec_rna['delta_protein'].dropna(), inc_rna['delta_protein'].dropna(), alternative='two-sided')
        print(f"\nMann-Whitney U Test (Comparison of Protein Delta between RNA-decreased and RNA-increased groups):")
        print(f"  U-statistic = {u_stat:.1f}")
        print(f"  p-value     = {p_val:.2e}")
        if p_val < 0.05:
            print("  -> The difference in protein delta distribution between the two groups is STATISTICALLY SIGNIFICANT (p < 0.05).")
            # Determine direction
            med_dec = dec_rna['delta_protein'].median()
            med_inc = inc_rna['delta_protein'].median()
            if med_dec > med_inc:
                print(f"  -> Protein delta is higher (more positive/rises more) when RNA half-life decreases (median: {med_dec:+.4f} vs {med_inc:+.4f}).")
            else:
                print(f"  -> Protein delta is lower (sinks more) when RNA half-life decreases (median: {med_dec:+.4f} vs {med_inc:+.4f}).")
        else:
            print("  -> The difference in protein delta distribution is NOT statistically significant (p >= 0.05).")

def generate_plots(df, output_dir):
    """Generates and saves correlation plots."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Map colors for clinical significance
    palette = {"Pathogenic": "#E74C3C", "Benign": "#2ECC71", "Uncertain significance": "#F1C40F", "Other": "#95A5A6"}
    
    # Build a color list based on unique values present to prevent seaborn errors
    available_categories = df['clinical_significance'].dropna().unique()
    plot_palette = {cat: palette.get(cat, "#34495E") for cat in available_categories}
    
    # ----------------------------------------------------
    # Plot 1: Scatter of delta_rna vs delta_protein
    # ----------------------------------------------------
    plt.figure(figsize=(10, 8))
    df_clean = df.dropna(subset=['delta_rna', 'delta_protein'])
    
    if len(df_clean) > 0:
        sns.scatterplot(
            data=df_clean,
            x='delta_rna',
            y='delta_protein',
            hue='clinical_significance',
            palette=plot_palette,
            alpha=0.7,
            s=40
        )
        
        # Calculate stats for textbox
        s_r, p_val_s = spearmanr(df_clean['delta_rna'], df_clean['delta_protein'])
        p_r, p_val_p = pearsonr(df_clean['delta_rna'], df_clean['delta_protein'])
        stats_text = (
            f"N = {len(df_clean)}\n"
            f"Spearman $\\rho$ = {s_r:+.3f} (p={p_val_s:.1e})\n"
            f"Pearson $r$ = {p_r:+.3f} (p={p_val_p:.1e})"
        )
        props = dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9', edgecolor='#BDC3C7', alpha=0.9)
        plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
                       verticalalignment='top', bbox=props)
                       
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
        plt.xlabel("RNA Delta (Mutated - WT)")
        plt.ylabel("Protein Delta (Mutated - WT)")
        plt.title("Correlation of Half-Life Changes (Delta) upon Mutation\nRNA vs Protein", fontsize=14, fontweight='bold')
        plt.legend(title="Clinical Significance", loc="upper right")
        
        scatter_path = os.path.join(output_dir, "rna_protein_delta_scatter.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to: {scatter_path}")
    else:
        print("No valid delta data points to plot scatter plot.")
    plt.close()
    
    # ----------------------------------------------------
    # Plot 2: Boxplot of delta_protein grouped by RNA change
    # ----------------------------------------------------
    df_box = df.dropna(subset=['delta_rna', 'delta_protein']).copy()
    if len(df_box) > 0:
        # Create category column
        df_box['rna_change'] = df_box['delta_rna'].apply(
            lambda x: 'RNA Decreased (<0)' if x < 0 else ('RNA Increased (>0)' if x > 0 else 'RNA Unchanged (=0)')
        )
        
        # Filter to only increased/decreased for cleaner visual if both exist
        df_box_filtered = df_box[df_box['rna_change'].isin(['RNA Decreased (<0)', 'RNA Increased (>0)'])]
        
        if len(df_box_filtered) > 0:
            plt.figure(figsize=(10, 7))
            
            # Combine categories of clinical significance that exist
            sns.boxplot(
                data=df_box_filtered,
                x='rna_change',
                y='delta_protein',
                hue='clinical_significance',
                palette=plot_palette,
                width=0.6
            )
            
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            plt.xlabel("RNA Half-Life Change Category")
            plt.ylabel("Protein Delta (Mutated - WT)")
            plt.title("Protein Half-Life Change by RNA Change Category", fontsize=14, fontweight='bold')
            plt.legend(title="Clinical Significance")
            
            boxplot_path = os.path.join(output_dir, "protein_delta_by_rna_change_boxplot.png")
            plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
            print(f"Saved boxplot to: {boxplot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Correlate RNA and Protein mutation half-life predictions.")
    parser.add_argument(
        "--rna_csv", 
        type=str, 
        default="/beegfs/prj/RNA_NLP/RNA_half_lives/saluki_prj_results/results/processed_mutation_results.csv",
        help="Pfad zur RNA mutation results CSV (Saluki predictions)"
    )
    parser.add_argument(
        "--protein_csv", 
        type=str, 
        default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_output/variant_predictions/processed_mutation_results.csv",
        help="Pfad zur Protein mutation results CSV (processed_mutation_results.csv)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_output/correlation_analysis",
        help="Ausgabeverzeichnis für die Ergebnisse"
    )
    
    args = parser.parse_args()
    
    # Verify input files exist
    inputs_ok = True
    for path_name, path_val in [("RNA CSV", args.rna_csv), ("Protein CSV", args.protein_csv)]:
        if not os.path.exists(path_val):
            print(f"Fehler: Die Datei '{path_name}' unter '{path_val}' existiert nicht.")
            inputs_ok = False
            
    if not inputs_ok:
        print("\nAbbruch: Ein oder mehrere benötigte Pfade wurden nicht gefunden.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Redirect stdout to both console and a text file in output_dir
    log_file_path = os.path.join(args.output_dir, "correlation_results.txt")
    logger = DualLogger(log_file_path)
    sys.stdout = logger
    
    # ----------------------------------------------------
    # 1. Load RNA mutation predictions
    # ----------------------------------------------------
    print(f"\nLade RNA Mutierte Vorhersagen von: {args.rna_csv}")
    rna_df = pd.read_csv(args.rna_csv, sep='\t')
    print(f"RNA-Datensatz geladen: {len(rna_df)} Zeilen")
    
    # Validate columns
    required_rna_cols = ['clinvar_id', 'pred_mut_mean', 'pred_wt']
    for col in required_rna_cols:
        if col not in rna_df.columns:
            raise KeyError(f"Fehler: Die Spalte '{col}' fehlt in der RNA-CSV.")
            
    # Extract only required columns, rename pred_wt to avoid collision
    rna_df_clean = rna_df[required_rna_cols].copy()
    rna_df_clean = rna_df_clean.rename(columns={
        'pred_wt': 'pred_wt_rna',
        'pred_mut_mean': 'pred_mut_rna'
    })
    
    # Compute delta_rna
    rna_df_clean['delta_rna'] = rna_df_clean['pred_mut_rna'] - rna_df_clean['pred_wt_rna']
    
    # Debugging NAs in RNA data
    print("\n--- [DEBUG] RNA Input NA Check ---")
    print(f"Total rows in rna_df_clean: {len(rna_df_clean)}")
    for col in rna_df_clean.columns:
        na_count = rna_df_clean[col].isna().sum()
        print(f"  Column '{col}': {na_count} NAs ({na_count/len(rna_df_clean)*100:.2f}%)")
    if rna_df_clean.isna().any().any():
        print("  Rows with any NA in RNA data (first 5):")
        print(rna_df_clean[rna_df_clean.isna().any(axis=1)].head(5).to_string())
    
    # ----------------------------------------------------
    # 2. Load Protein predictions from processed_mutation_results.csv
    # ----------------------------------------------------
    print(f"\nLade Protein Vorhersagen von: {args.protein_csv}")
    protein_df = pd.read_csv(args.protein_csv)
    print(f"Protein-Datensatz geladen: {len(protein_df)} Zeilen")
    
    required_prot_cols = ['clinvar_id', 'clinical_significance', 'tid', 'gene', 'pred_wt', 'pred_mut_mean', 'delta_halflife']
    for col in required_prot_cols:
        if col not in protein_df.columns:
            raise KeyError(f"Fehler: Die Spalte '{col}' fehlt in der Protein-CSV.")
            
    # Extract, rename columns to align with previous merged data format
    protein_df_clean = protein_df[required_prot_cols].copy()
    protein_df_clean = protein_df_clean.rename(columns={
        'pred_wt': 'pred_wt_protein',
        'pred_mut_mean': 'pred_mut_protein',
        'delta_halflife': 'delta_protein'
    })
    
    # Debugging NAs in Protein data
    print("\n--- [DEBUG] Protein Input NA Check ---")
    print(f"Total rows in protein_df_clean: {len(protein_df_clean)}")
    for col in protein_df_clean.columns:
        na_count = protein_df_clean[col].isna().sum()
        print(f"  Column '{col}': {na_count} NAs ({na_count/len(protein_df_clean)*100:.2f}%)")
    if protein_df_clean.isna().any().any():
        print("  Rows with any NA in Protein data (first 5):")
        print(protein_df_clean[protein_df_clean.isna().any(axis=1)].head(5).to_string())
    
    # ----------------------------------------------------
    # 3. Merge Datasets
    # ----------------------------------------------------
    print("\nFühre Merges durch...")
    # Merge RNA and Protein predictions on clinvar_id
    df_merged = pd.merge(rna_df_clean, protein_df_clean, on='clinvar_id', how='inner')
    print(f"Gemerged auf 'clinvar_id' (inner join): {len(df_merged)} Zeilen")
    
    if len(df_merged) == 0:
        print("Abbruch: Keine übereinstimmenden ClinVar-IDs gefunden.")
        return
        
    # Reorder columns for clean presentation
    final_cols = [
        'clinvar_id', 'clinical_significance', 'tid', 'gene',
        'pred_wt_rna', 'pred_mut_rna', 'delta_rna',
        'pred_wt_protein', 'pred_mut_protein', 'delta_protein'
    ]
    # Filter to only columns that actually exist (handling any edge cases)
    final_cols = [c for c in final_cols if c in df_merged.columns]
    df_final = df_merged[final_cols].copy()
    
    # Debugging NAs in Final output DataFrame
    print("\n--- [DEBUG] Final Output NA Check ---")
    print(f"Total rows in df_final: {len(df_final)}")
    for col in df_final.columns:
        na_count = df_final[col].isna().sum()
        print(f"  Column '{col}': {na_count} NAs ({na_count/len(df_final)*100:.2f}%)")
    
    # Show example rows with NAs in the final output
    na_rows = df_final[df_final.isna().any(axis=1)]
    if len(na_rows) > 0:
        print(f"  Total rows with at least one NA in df_final: {len(na_rows)} ({len(na_rows)/len(df_final)*100:.2f}%)")
        print("  Showing first 10 rows with NAs in the final merged data:")
        print(na_rows.head(10).to_string())
    else:
        print("  No NAs found in the final merged DataFrame!")
        
    # Save to CSV
    out_csv_path = os.path.join(args.output_dir, "rna_protein_merged_predictions.csv")
    df_final.to_csv(out_csv_path, index=False)
    print(f"\nGemergeder DataFrame als CSV gespeichert unter: {out_csv_path}")
    
    # ----------------------------------------------------
    # 5. Correlation & Statistical Analysis
    # ----------------------------------------------------
    print("\n" + "="*60)
    print("KORRELATIONSANALYSE (Gesamt und nach clinical_significance)")
    print("="*60)
    
    # 5a. Mutated predictions correlation (pred_mut_rna vs pred_mut_protein)
    print("\n1. Korrelation der vorhergesagten Mutations-Halbwertszeiten:")
    calculate_correlations(df_final, 'pred_mut_rna', 'pred_mut_protein', label="Gesamt (mutiert)")
    
    # Grouped by significance
    for sig in df_final['clinical_significance'].dropna().unique():
        df_sig = df_final[df_final['clinical_significance'] == sig]
        calculate_correlations(df_sig, 'pred_mut_rna', 'pred_mut_protein', label=f"{sig} (mutiert)")
        
    # 5b. Delta predictions correlation (delta_rna vs delta_protein)
    if 'delta_protein' in df_final.columns and not df_final['delta_protein'].isna().all():
        print("\n2. Korrelation der Halbwertszeiten-Änderungen (Delta: Mutiert - WT):")
        calculate_correlations(df_final, 'delta_rna', 'delta_protein', label="Gesamt (delta)")
        
        for sig in df_final['clinical_significance'].dropna().unique():
            df_sig = df_final[df_final['clinical_significance'] == sig]
            calculate_correlations(df_sig, 'delta_rna', 'delta_protein', label=f"{sig} (delta)")
            
        # 5c. Specific Hypothesis Test
        analyze_hypothesis(df_final)
        
        # 5d. Hypothesis test by clinical significance
        for sig in df_final['clinical_significance'].dropna().unique():
            df_sig = df_final[df_final['clinical_significance'] == sig]
            if len(df_sig) >= 10:
                print(f"\n--- Hypothesen-Test nur für: {sig} ---")
                analyze_hypothesis(df_sig)
    else:
        print("\nKorrelation der Delta-Werte und Hypothesen-Test übersprungen, da 'delta_protein' nicht berechnet wurde.")
        
    # ----------------------------------------------------
    # 6. Generate Visualizations
    # ----------------------------------------------------
    if 'delta_protein' in df_final.columns and not df_final['delta_protein'].isna().all():
        print("\nGeneriere Abbildungen...")
        try:
            generate_plots(df_final, args.output_dir)
        except Exception as e:
            print(f"Warnung: Abbildungen konnten nicht erzeugt werden. Fehler: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nAnalyse erfolgreich abgeschlossen!")
    
    # Restore stdout and close logger
    sys.stdout = logger.terminal
    logger.close()
    print(f"\nTextanalyse wurde unter '{log_file_path}' gespeichert.")

if __name__ == "__main__":
    main()
