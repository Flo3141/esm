import os
import argparse
import pandas as pd
import torch
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
        self.data = data
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

def main():
    parser = argparse.ArgumentParser(description="Train a Regression Head on ESM for Protein Half-life")
    parser.add_argument("--csv_path", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/Protein_half_lifes.csv", help="Pfad zur CSV-Datei")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t12_35M_UR50D", help="ESM Modellname von Hugging Face")
    parser.add_argument("--cache_dir", type=str, default="/beegfs/prj/RNA_NLP/protein_half_lives/esm_weights", help="Speicherort für Hugging Face Gewichte")
    parser.add_argument("--epochs", type=int, default=3, help="Anzahl der Trainingsepochen")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch Größe")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Lernrate für den Head")
    parser.add_argument("--fold", type=int, default=0, help="Welcher Fold (0-3) trainiert werden soll.")
    
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
        evaluation_strategy="epoch",
        logging_dir="/beegfs/prj/RNA_NLP/protein_half_lives/esm_logs",
        logging_steps=10,
        learning_rate=args.learning_rate,
        # Deaktiviere Checkpointing, um nicht das komplette Modell zu speichern
        save_strategy="no" 
    )

    head_weights_path = os.path.join(args.cache_dir, f"regression_head_weights_fold_{args.fold}.pt")
    save_callback = SaveBestHeadCallback(head_weights_path)

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

if __name__ == "__main__":
    main()
