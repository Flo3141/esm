import os
import argparse
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmForSequenceClassification
import gc

class ProteinMutationInferenceDataset(Dataset):
    """Dataset class for protein sequence inference."""
    def __init__(self, df, tokenizer, max_length=1024):
        self.data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = str(row['mutated_AA'])

        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        return item

def main():
    parser = argparse.ArgumentParser(description="Predict mutated sequence half-lives using ESM fold models.")
    parser.add_argument("--csv_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_data/Protein_half_lifes_mutated.csv", help="Pfad zur Protein_half_lifes_mutated.csv")
    parser.add_argument("--output_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_output/Protein_half_lifes_predicted.csv", help="Pfad zur Ausgabedatei")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t12_35M_UR50D", help="ESM Modellname von Hugging Face")
    parser.add_argument("--cache_dir", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_weights", help="Speicherort für Hugging Face Gewichte")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Größe für die Vorhersage")
    parser.add_argument("--folds", type=str, default="0,1,2,3", help="Komma-separierte Fold-Indizes, die verwendet werden sollen")
    
    args = parser.parse_args()

    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    os.environ['HF_HOME'] = args.cache_dir

    print(f"Lade Tokenizer: {args.model_name}")
    tokenizer = EsmTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    print(f"Lade Eingabedaten von: {args.csv_path}")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Fehler: {args.csv_path} existiert nicht.")
    df = pd.read_csv(args.csv_path)
    
    # Check if mutated_AA column exists
    if "mutated_AA" not in df.columns:
        raise KeyError("Fehler: Die Spalte 'mutated_AA' fehlt in der Eingabe-CSV.")

    print(f"Anzahl Zeilen für Vorhersage: {len(df)}")
    
    # Parse fold indices
    try:
        fold_indices = [int(f.strip()) for f in args.folds.split(",")]
    except ValueError:
        raise ValueError("Folds müssen eine komma-separierte Liste von Integern sein, z.B. 0,1,2,3")

    # Filter to only folds that actually have weight files
    valid_folds = []
    weight_paths = {}
    for f_idx in fold_indices:
        weights_path = os.path.join(args.cache_dir, f"regression_head_weights_fold_{f_idx}.pt")
        if os.path.exists(weights_path):
            valid_folds.append(f_idx)
            weight_paths[f_idx] = weights_path
        else:
            print(f"Warnung: Keine Gewichte für Fold {f_idx} unter {weights_path} gefunden. Überspringe...")

    if not valid_folds:
        raise FileNotFoundError("Fehler: Für keinen der angegebenen Folds wurden Gewichtsdateien gefunden!")
    
    print(f"Folgende Folds werden für die Ensemblierung verwendet: {valid_folds}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Verwende Gerät: {device}")

    # Initialize model once
    print(f"Lade Basismodell: {args.model_name}")
    model = EsmForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=1, 
        cache_dir=args.cache_dir
    )
    model = model.to(device)
    model.eval()

    # Create dataset & loader
    dataset = ProteinMutationInferenceDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Dictionary to store predictions per fold
    fold_predictions = {}

    for f_idx in valid_folds:
        print(f"\n--- Starte Vorhersage für Fold {f_idx} ---")
        weights_path = weight_paths[f_idx]
        print(f"Lade Head-Gewichte von: {weights_path}")
        
        # Load weights into classifier
        state_dict = torch.load(weights_path, map_location=device)
        model.classifier.load_state_dict(state_dict)
        
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
                    
        fold_predictions[f_idx] = predictions
        
        # Clean up
        del state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\nBerechne Ensemblemittelwert über alle Folds...")
    # Convert predictions to numpy array (shape: num_folds, num_samples)
    preds_arr = np.array([fold_predictions[f] for f in valid_folds])
    mean_preds = np.mean(preds_arr, axis=0)

    # Add prediction to dataframe
    df['pred_mut_halflife'] = mean_preds

    # Select requested columns
    output_cols = ['tid', 'gene', 'clinvar_id', 'clinical_significance', 'halflife', 'pred_mut_halflife']
    for col in output_cols:
        if col not in df.columns:
            raise KeyError(f"Fehler: Die benötigte Spalte '{col}' existiert nicht in der Eingabe-CSV.")

    df_out = df[output_cols]

    # Save to output path
    print(f"Speichere Ergebnisse nach: {args.output_path}")
    df_out.to_csv(args.output_path, index=False)
    print("Fertig!")

if __name__ == "__main__":
    main()
