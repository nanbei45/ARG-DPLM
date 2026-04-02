import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
import esm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, matthews_corrcoef, accuracy_score)


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

esm_dir = '/data0/zhangpinglu/esm/ESM-2/esm2'
os.makedirs(esm_dir, exist_ok=True)
os.environ['TORCH_HOME'] = esm_dir
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################
# Configuration parameters
########################
class Config:

    train_pos_seqs_file = '/data0/zhangpinglu/esm/ESM-2/data/positive/train.fasta'
    train_neg_seqs_file = '/data0/zhangpinglu/esm/ESM-2/data/negative/aa_neg_train.fasta'
    val_pos_seqs_file = '/data0/zhangpinglu/esm/ESM-2/data/positive/val.fasta'
    val_neg_seqs_file = '/data0/zhangpinglu/esm/ESM-2/data/negative/aa_neg_val.fasta'
    test_pos_seqs_file = '/data0/zhangpinglu/esm/ESM-2/data/positive/test.fasta'
    test_neg_seqs_file = '/data0/zhangpinglu/esm/ESM-2/data/negative/aa_neg_test.fasta'

    # model parameters
    esm_model_name = "esm2_t33_650M_UR50D"
    t5_model_path = "/data0/zhangpinglu/esm/ESM-2/huggingface/prot_t5_xl_uniref50"
    max_length = 512
    batch_size = 64
    accum_steps = 2
    epochs = 200
    lr = 1e-4
    model_lr = 1e-5
    early_stop_patience = 20
    metrics_file = "/data0/zhangpinglu/esm/ESM-2/training_metrics2.csv"
    best_model_path = "/data0/zhangpinglu/esm/ESM-2/ARG-DPLM-MODEL.pth"

# Fine-tuning model definition
class FineTuneModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Load the original large model
        self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet(Config.esm_model_name)
        self.esm_converter = self.esm_alphabet.get_batch_converter()
        self.t5_tokenizer = T5Tokenizer.from_pretrained(Config.t5_model_path, do_lower_case=False)
        self.t5_model = T5EncoderModel.from_pretrained(Config.t5_model_path)

        # Freeze parameters
        for param in self.esm_model.parameters():
            param.requires_grad = False
        for param in self.t5_model.parameters():
            param.requires_grad = False

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280 + 1024, 512),  # ESM(1280) + T5(1024)
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 1)
        )

        # Thaw the last 3 layers
        self._unfreeze_layers()

    def _unfreeze_layers(self):
        # Thaw the last 3 layers of ESM
        for layer in self.esm_model.layers[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Thaw the last two layers of T5
        for layer in self.t5_model.encoder.block[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, raw_sequences):
        # ESM Feature Extraction
        esm_data = [(str(i), seq[:Config.max_length]) for i, seq in enumerate(raw_sequences)]
        _, _, esm_tokens = self.esm_converter(esm_data)
        esm_tokens = esm_tokens.to(device)
        esm_output = self.esm_model(esm_tokens, repr_layers=[33])["representations"][33].mean(1)

        # T5 Feature Extraction
        t5_inputs = self.t5_tokenizer(
            raw_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.max_length,
            return_attention_mask=True
        ).to(device)
        t5_output = self.t5_model(**t5_inputs).last_hidden_state.mean(1)

        # Feature fusion
        fused = torch.cat([esm_output, t5_output], dim=1)
        return torch.sigmoid(self.classifier(fused)).squeeze()


class ARGDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]



def clean_sequence(seq):
    """
    Clean the protein sequence and replace all non-standard amino acids with X.
    :param seq: Original amino acid sequence
    :return: Cleaned sequence
    """
    valid_chars = set("ACDEFGHIKLMNPQRSTVWYBXZOU-")

    ambiguous_map = {
        'J': 'X',
        'B': 'X',
        'Z': 'X',
        'U': 'X',
        'O': 'X',
    }


    for amb_char, replacement in ambiguous_map.items():
        seq = seq.replace(amb_char, replacement)

    cleaned = ''.join(aa if aa in valid_chars else 'X' for aa in seq)

    return cleaned
def main():
    train_pos_seqs = [clean_sequence(str(rec.seq)) for rec in SeqIO.parse(Config.train_pos_seqs_file, "fasta")]
    train_neg_seqs = [clean_sequence(str(rec.seq)) for rec in SeqIO.parse(Config.train_neg_seqs_file, "fasta")]
    val_pos_seqs = [clean_sequence(str(rec.seq)) for rec in SeqIO.parse(Config.val_pos_seqs_file, "fasta")]
    val_neg_seqs = [clean_sequence(str(rec.seq)) for rec in SeqIO.parse(Config.val_neg_seqs_file, "fasta")]
    test_pos_seqs = [clean_sequence(str(rec.seq)) for rec in SeqIO.parse(Config.test_pos_seqs_file, "fasta")]
    test_neg_seqs = [clean_sequence(str(rec.seq)) for rec in SeqIO.parse(Config.test_neg_seqs_file, "fasta")]

    train_labels = [1] * len(train_pos_seqs) + [0] * len(train_neg_seqs)
    val_labels = [1] * len(val_pos_seqs) + [0] * len(val_neg_seqs)
    test_labels = [1] * len(test_pos_seqs) + [0] * len(test_neg_seqs)

    train_dataset = ARGDataset(train_pos_seqs + train_neg_seqs, train_labels)
    test_dataset = ARGDataset(test_pos_seqs + test_neg_seqs, test_labels)
    val_dataset = ARGDataset(val_pos_seqs + val_neg_seqs, val_labels)

    model = FineTuneModel().to(device)
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.AdamW([
        {'params': model.esm_model.parameters(), 'lr': Config.model_lr},
        {'params': model.t5_model.parameters(), 'lr': Config.model_lr},
        {'params': model.classifier.parameters(), 'lr': Config.lr}
    ])

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: (
            [item[0] for item in batch],  # raw_sequences
            torch.tensor([item[1] for item in batch], dtype=torch.float)
        )
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.tensor([item[1] for item in batch], dtype=torch.float)
        )
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            torch.tensor([item[1] for item in batch], dtype=torch.float)
        )
    )

    early_stop_counter = 0

    best_acc = 0.0

    csv_header = ["epoch", "timestamp", "train_loss", "val_loss",
                  "accuracy", "precision", "recall", "f1",
                  "auc", "mcc"]

    with open(Config.metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (seqs, labels) in enumerate(tqdm(train_loader)):
            labels = labels.to(device)

            outputs = model(seqs)
            loss = criterion(outputs, labels)

            loss = loss / Config.accum_steps
            loss.backward()
            if (batch_idx + 1) % Config.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item() * Config.accum_steps

        model.eval()
        all_labels = []
        all_probs = []
        total_val_loss = 0.0
        with torch.no_grad():
            for seqs, labels in val_loader:
                labels = labels.to(device)
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())

        val_loss = total_val_loss / len(test_dataset)
        all_preds = [1 if p > 0.5 else 0 for p in all_probs]
        metrics = {
            'loss': val_loss,
            'acc': accuracy_score(all_labels, all_preds),
            'pre': precision_score(all_labels, all_preds, zero_division=0),
            'rec': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs),
            'mcc': matthews_corrcoef(all_labels, all_preds)
        }

        log_data = {
            "epoch": epoch + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_loss": round(total_loss / len(train_loader), 4),
            "val_loss": round(metrics['loss'], 4),
            "accuracy": round(metrics['acc'], 4),
            "precision": round(metrics['pre'], 4),
            "recall": round(metrics['rec'], 4),
            "f1": round(metrics['f1'], 4),
            "auc": round(metrics['auc'], 4),
            "mcc": round(metrics['mcc'], 4),

        }

        with open(Config.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(log_data.values())

        print(f"\nEpoch {epoch + 1}/{Config.epochs}")
        print(f"Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"Val Loss: {metrics['loss']:.4f} | Acc: {metrics['acc']:.4f}")
        print(f"Pre: {metrics['pre']:.4f} | Rec: {metrics['rec']:.4f}")
        print(f"F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")

        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            early_stop_counter = 0
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'metrics': metrics
            }, Config.best_model_path)
            print("↻ Saved best model!")
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{Config.early_stop_patience}")
            if early_stop_counter >= Config.early_stop_patience:
                print("⏹ Early stopping triggered!")
                break


    checkpoint = torch.load('/data0/zhangpinglu/esm/ESM-2/ARG-DPLM-MODEL.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    all_labels = []
    all_probs = []
    total_val_loss = 0.0
    with torch.no_grad():
        for seqs, labels in test_loader:
            labels = labels.to(device)
            outputs = model(seqs)

            loss = criterion(outputs, labels)
            total_val_loss += loss.item() * labels.size(0)


            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())


    val_loss = total_val_loss / len(test_dataset)
    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    metrics = {
        'loss': val_loss,
        'acc': accuracy_score(all_labels, all_preds),
        'pre': precision_score(all_labels, all_preds, zero_division=0),
        'rec': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs),
        'mcc': matthews_corrcoef(all_labels, all_preds)
    }

    log_data = {
        "epoch": 9999,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_loss": 0,
        "val_loss": round(metrics['loss'], 4),
        "accuracy": round(metrics['acc'], 4),
        "precision": round(metrics['pre'], 4),
        "recall": round(metrics['rec'], 4),
        "f1": round(metrics['f1'], 4),
        "auc": round(metrics['auc'], 4),
        "mcc": round(metrics['mcc'], 4),

    }
    with open(Config.metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(log_data.values())

    print(f"\nTest set evaluation results")
    print(f"Val Loss: {metrics['loss']:.4f} | Acc: {metrics['acc']:.4f}")
    print(f"Pre: {metrics['pre']:.4f} | Rec: {metrics['rec']:.4f}")
    print(f"F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")



if __name__ == "__main__":
    main()