import os
import argparse
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import EsmTokenizer, EsmForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class SaveBestHeadCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_loss = float('inf')

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get('eval_loss', None)
        if eval_loss is not None and eval_loss < self.best_loss:
            self.best_loss = eval_loss
            print(f"\nNeue beste Validation Loss: {eval_loss:.4f}. Speichere Head-Gewichte nach {self.save_path}...")
            model = kwargs['model']
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.classifier.state_dict(), self.save_path)

class ProteinHalfLifeDataset(Dataset):
    def __init__(self, csv_file, tokenizer, splits=None, max_length=1024):
        data = pd.read_csv(csv_file)
        if splits is not None:
            data = data[data['split'].isin(splits)]
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['AA']
        halflife = float(row['halflife'])

        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(halflife, dtype=torch.float32)

        return item

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = predictions.flatten()
    labels = labels.flatten()

    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)

    return {"mse": mse, "mae": mae, "r2": r2}

def evaluate_predictions(checkpoint_dir):
    res_text = ""
    test_pred = os.path.join(checkpoint_dir, "test_predictions.csv")
    if not os.path.exists(test_pred):
        print(f"Fehler: {test_pred} existiert nicht.")
        return
    df = pd.read_csv(test_pred)
    df_eval = df.drop(columns=['sequence'])
    predict_labels = df_eval["prediction"]
    true_labels = df_eval["label"]
    all_mse = np.mean((np.array(predict_labels) - np.array(true_labels)) ** 2)
    all_prearson = df_eval.corr(method='pearson').iloc[0, 1]
    all_spearman = df_eval.corr(method='spearman').iloc[0, 1]
    res_text = f"----- ESM Evaluation -----\n"
    res_text += f"MSE: {all_mse}\n"
    res_text += f"Pearson: {all_prearson}\n"
    res_text += f"Spearman: {all_spearman}\n"
    
    print(res_text)
    with open(os.path.join(checkpoint_dir, "esm_results.txt"), "w") as f:
        f.write(res_text)

def evaluate_on_test(args, tokenizer):
    print(f"Starte Evaluierung auf Test-Splits [8, 9] für Fold {args.fold}...")
    
    model = EsmForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=1, 
        cache_dir=args.cache_dir
    )
    
    head_weights_path = os.path.join(args.cache_dir, f"regression_head_weights_fold_{args.fold}.pt")
    if not os.path.exists(head_weights_path):
        raise FileNotFoundError(f"Keine trainierten Gewichte unter {head_weights_path} gefunden!")
        
    print(f"Lade beste Gewichte des Heads von {head_weights_path}...")
    model.classifier.load_state_dict(torch.load(head_weights_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_dataset = ProteinHalfLifeDataset(args.csv_path, tokenizer, splits=[8, 9])
    print(f"Anzahl Testbeispiele: {len(test_dataset)}")
    
    if len(test_dataset) == 0:
        print("Keine Testdaten vorhanden!")
        return
        
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1).cpu().numpy()
            
            if logits.ndim == 0:
                predictions.append(float(logits))
            else:
                predictions.extend(logits.tolist())
            labels.extend(batch['labels'].numpy().tolist())
            
    sequences = test_dataset.data['AA'].tolist()
    
    df_pred = pd.DataFrame({
        "sequence": sequences,
        "prediction": predictions,
        "label": labels
    })
    
    test_pred_path = os.path.join(args.cache_dir, "test_predictions.csv")
    df_pred.to_csv(test_pred_path, index=False)
    print(f"Test-Vorhersagen gespeichert unter {test_pred_path}")
    
    evaluate_predictions(args.cache_dir)

def main():
    parser = argparse.ArgumentParser(description="Train a Regression Head on ESM for Protein Half-life")
    parser.add_argument("--csv_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/Protein_half_lifes.csv", help="Pfad zur CSV-Datei")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t12_35M_UR50D", help="ESM Modellname von Hugging Face")
    parser.add_argument("--cache_dir", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_weights", help="Speicherort für Hugging Face Gewichte")
    parser.add_argument("--epochs", type=int, default=3, help="Anzahl der Trainingsepochen")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Größe")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Lernrate für den Head")
    parser.add_argument("--fold", type=int, default=0, help="Welcher Fold (0-3) trainiert werden soll.")
    parser.add_argument("--only_eval", action="store_true", help="Führe nur die Evaluierung auf den Testdaten aus.")
    
    args = parser.parse_args()

    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    os.environ['HF_HOME'] = args.cache_dir

    print(f"Lade Tokenizer: {args.model_name}")
    tokenizer = EsmTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    print(f"Lade Datensatz: {args.csv_path}")
    
    folds = {
        0: {'train': [0, 1, 2, 3, 4, 5], 'val': [6, 7]},
        1: {'train': [0, 1, 2, 3, 6, 7], 'val': [4, 5]},
        2: {'train': [0, 1, 4, 5, 6, 7], 'val': [2, 3]},
        3: {'train': [2, 3, 4, 5, 6, 7], 'val': [0, 1]}
    }
    
    if args.fold not in folds:
        raise ValueError("Fold muss zwischen 0 und 3 liegen.")

    train_splits = folds[args.fold]['train']
    val_splits = folds[args.fold]['val']

    train_dataset = ProteinHalfLifeDataset(args.csv_path, tokenizer, splits=train_splits)
    eval_dataset = ProteinHalfLifeDataset(args.csv_path, tokenizer, splits=val_splits)
    print(f"Trainingsbeispiele: {len(train_dataset)}, Validierungsbeispiele: {len(eval_dataset)}")

    print(f"Lade Modell: {args.model_name}")
    # num_labels=1 führt dazu, dass ein Regression Head (linearer Layer) erzeugt wird und MSE Loss verwendet wird.
    model = EsmForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=1, 
        cache_dir=args.cache_dir
    )

    print("Friere Basis-Modell-Gewichte ein...")
    # Alle Parameter des ESM-Encoders einfrieren
    for param in model.esm.parameters():
        param.requires_grad = False

    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainierbare Parameter: {trainable_params}")

    training_args = TrainingArguments(
        output_dir="/beegfs/prj/RNA_NLP/protein_half_lives/esm_output",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        logging_dir="/beegfs/prj/RNA_NLP/protein_half_lives/esm_logs",
        logging_steps=10,
        learning_rate=args.learning_rate,
        # Deaktiviere Checkpointing, um nicht das komplette Modell zu speichern
        save_strategy="no" 
    )
    
    head_weights_path = os.path.join(args.cache_dir, f"regression_head_weights_fold_{args.fold}.pt")
    save_callback = SaveBestHeadCallback(head_weights_path)

    if not args.only_eval:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[save_callback]
        )

        print(f"Starte Training für Fold {args.fold}...")
        trainer.train()
        print("Training abgeschlossen! Die besten Gewichte wurden bereits basierend auf dem Validation Loss gespeichert.")
    else:
        print("Überspringe Training. Starte nur Evaluierung...")

    evaluate_on_test(args, tokenizer)

if __name__ == "__main__":
    main()
