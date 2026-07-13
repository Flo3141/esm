import csv
import gzip
import re
import os
import time
import argparse

# Mapping of 3-letter amino acid codes to 1-letter codes
AMINO_ACIDS = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*'  # Stop codon
}

def load_transcripts(csv_path):
    """
    Loads transcripts from the mapping CSV, groups them by gene symbol,
    and returns the grouping dictionary and the original fieldnames.
    """
    transcripts_by_gene = {}
    fieldnames = []
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return transcripts_by_gene, fieldnames

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            gene_symbol = row.get("gene_symbol")
            if gene_symbol:
                # Group by gene symbol for quick lookup
                if gene_symbol not in transcripts_by_gene:
                    transcripts_by_gene[gene_symbol] = []
                transcripts_by_gene[gene_symbol].append(row)
    
    print(f"Loaded transcripts for {len(transcripts_by_gene)} unique gene symbols.")
    return transcripts_by_gene, fieldnames

def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Mutate sequences based on variants.")
    parser.add_argument("--csv_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_data/ensembl_gene_mapping.csv", help="Path to the gene mapping CSV file")
    parser.add_argument("--variant_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_data/variant_summary.txt.gz", help="Path to the variant summary txt.gz file")
    parser.add_argument("--output_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_data/Protein_half_lifes_mutated.csv", help="Path to save the mutated sequences CSV file")
    args = parser.parse_args()

    csv_path = args.csv_path
    variant_path = args.variant_path
    output_path = args.output_path
    
    print("1. Loading transcripts mapping...")
    transcripts_by_gene, original_fieldnames = load_transcripts(csv_path)
    
    if not transcripts_by_gene:
        print("No transcripts loaded. Exiting.")
        return
        
    print(f"\n2. Setting up output CSV: {output_path}...")
    output_fieldnames = original_fieldnames + [
        "clinvar_id", "clinical_significance", "mutated_AA", "phenotype", 
        "variant_type", "mutation_type"
    ]
    
    # Regular expression to extract p.XxxPosYyy
    mutation_regex = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')
    
    total_written = 0
    total_scanned_variants = 0
    
    with open(output_path, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames)
        writer.writeheader()
        
        print(f"\n3. Scanning {variant_path} and writing mutations...")
        with gzip.open(variant_path, 'rt', encoding='utf-8') as f:
            # Read header
            header = f.readline().strip().split('\t')
            gene_idx = header.index("GeneSymbol")
            name_idx = header.index("Name")
            assembly_idx = header.index("Assembly")
            sig_idx = header.index("ClinicalSignificance")
            variation_id_idx = header.index("VariationID")
            phenotype_idx = header.index("PhenotypeList")
            type_idx = header.index("Type")
            
            for line in f:
                total_scanned_variants += 1
                cols = line.strip().split('\t')
                if len(cols) <= max(gene_idx, name_idx, assembly_idx, sig_idx, variation_id_idx, phenotype_idx, type_idx):
                    continue
                    
                # 1. Filter for GRCh38 assembly
                if cols[assembly_idx] != "GRCh38":
                    continue
                    
                # 2. Filter for cardiovascular phenotype keyword
                if "cardiovascular" not in cols[phenotype_idx].lower():
                    continue
                    
                gene_symbol = cols[gene_idx]
                
                # 3. Check if this gene symbol exists in our transcripts mapping (Fast lookup)
                if gene_symbol not in transcripts_by_gene:
                    continue
                    
                # 4. Filter for Pathogenic or Benign (exclude conflicting or uncertain)
                sig = cols[sig_idx].lower()
                if "conflicting" in sig or "uncertain significance" in sig:
                    continue
                    
                is_pathogenic = "pathogenic" in sig
                is_benign = "benign" in sig
                if not (is_pathogenic or is_benign):
                    continue
                    
                # Determine classification
                classification = "Pathogenic" if is_pathogenic else "Benign"
                variant_id = cols[variation_id_idx]
                
                # 5. Extract protein mutation using regex
                variant_name = cols[name_idx]
                match = mutation_regex.search(variant_name)
                
                if match:
                    ref_3aa, pos_str, alt_3aa = match.groups()
                    pos = int(pos_str)
                    
                    ref_aa = AMINO_ACIDS.get(ref_3aa)
                    alt_aa = AMINO_ACIDS.get(alt_3aa)
                    
                    # Skip if we don't recognize the amino acids
                    if not ref_aa or not alt_aa:
                        continue
                        
                    # Check all transcripts for this gene
                    for row in transcripts_by_gene[gene_symbol]:
                        sequence = row.get("AA", "")
                        
                        # Validate position and amino acid
                        if len(sequence) >= pos:
                            actual_aa = sequence[pos - 1]
                            
                            if actual_aa == ref_aa:
                                # Apply mutation
                                if alt_aa == '*':
                                    # Nonsense mutation: truncate everything from the stop codon onwards
                                    mutated_sequence = sequence[:pos-1]
                                else:
                                    mutated_sequence = sequence[:pos-1] + alt_aa + sequence[pos:]
                                
                                # Construct the output row
                                new_row = dict(row)
                                new_row["clinvar_id"] = variant_id
                                new_row["clinical_significance"] = classification
                                new_row["mutated_AA"] = mutated_sequence
                                new_row["phenotype"] = cols[phenotype_idx]
                                new_row["variant_type"] = cols[type_idx]
                                new_row["mutation_type"] = "nonsense" if alt_aa == '*' else "missense"
                                
                                # Write to output CSV
                                writer.writerow(new_row)
                                total_written += 1
                                
                                if total_written % 10000 == 0:
                                    elapsed = time.time() - start_time
                                    print(f"  Written {total_written} mutated sequences... (Elapsed: {elapsed:.1f}s)")
                                    
    end_time = time.time()
    total_time = end_time - start_time
    print("\n" + "="*80)
    print("BATCH MUTATION PROCESSING COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Total variants scanned in file:   {total_scanned_variants}")
    print(f"Total mutated sequences written:   {total_written}")
    print(f"Output saved to:                   {output_path}")
    print(f"Total time elapsed:                {total_time:.2f} seconds")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()