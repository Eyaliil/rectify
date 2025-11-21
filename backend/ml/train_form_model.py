"""
Train Multi-Task Model for Exercise Classification + Form Quality
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
import argparse

from models.form_classifier import MultiTaskClassifier


class MultiTaskTrainer:
    """Trainer for multi-task exercise + form classification."""

    def __init__(self, data_dir="data/processed_form", batch_size=32, lr=0.001):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Using device: {self.device}")

        self.load_data()
        self.build_model()

    def load_data(self):
        """Load multi-task dataset."""
        X_train = np.load(self.data_dir / "X_train.npy")
        X_val = np.load(self.data_dir / "X_val.npy")
        X_test = np.load(self.data_dir / "X_test.npy")

        y_ex_train = np.load(self.data_dir / "y_exercise_train.npy")
        y_ex_val = np.load(self.data_dir / "y_exercise_val.npy")
        y_ex_test = np.load(self.data_dir / "y_exercise_test.npy")

        y_form_train = np.load(self.data_dir / "y_form_train.npy")
        y_form_val = np.load(self.data_dir / "y_form_val.npy")
        y_form_test = np.load(self.data_dir / "y_form_test.npy")

        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)

        # Create dataloaders
        self.train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_ex_train),
                torch.LongTensor(y_form_train)
            ),
            batch_size=self.batch_size, shuffle=True
        )

        self.val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_ex_val),
                torch.LongTensor(y_form_val)
            ),
            batch_size=self.batch_size
        )

        self.test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_test),
                torch.LongTensor(y_ex_test),
                torch.LongTensor(y_form_test)
            ),
            batch_size=self.batch_size
        )

        print(f"\n=== Dataset Loaded ===")
        print(f"Exercises: {list(self.metadata['exercise_label2id'].keys())}")
        print(f"Form classes: good, bad")
        print(f"Input shape: {X_train.shape[1:]}")
        print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    def build_model(self):
        """Build multi-task model."""
        self.model = MultiTaskClassifier(
            input_size=self.metadata['input_size'],
            hidden_size=128,
            num_layers=2,
            num_exercises=self.metadata['num_exercises'],
            num_form_classes=2,
            dropout=0.3
        ).to(self.device)

        # Losses
        self.exercise_criterion = nn.CrossEntropyLoss()
        self.form_criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel Parameters: {total_params:,}")

    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        ex_correct = 0
        form_correct = 0
        total = 0

        for X, y_ex, y_form in self.train_loader:
            X, y_ex, y_form = X.to(self.device), y_ex.to(self.device), y_form.to(self.device)

            self.optimizer.zero_grad()

            ex_logits, form_logits = self.model(X)

            # Multi-task loss (weighted sum)
            ex_loss = self.exercise_criterion(ex_logits, y_ex)
            form_loss = self.form_criterion(form_logits, y_form)
            loss = ex_loss + form_loss  # Equal weighting

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            ex_correct += (ex_logits.argmax(dim=1) == y_ex).sum().item()
            form_correct += (form_logits.argmax(dim=1) == y_form).sum().item()
            total += y_ex.size(0)

        return total_loss / len(self.train_loader), ex_correct / total, form_correct / total

    def validate(self, loader):
        """Validate on given loader."""
        self.model.eval()
        total_loss = 0
        ex_correct = 0
        form_correct = 0
        total = 0

        with torch.no_grad():
            for X, y_ex, y_form in loader:
                X, y_ex, y_form = X.to(self.device), y_ex.to(self.device), y_form.to(self.device)

                ex_logits, form_logits = self.model(X)

                ex_loss = self.exercise_criterion(ex_logits, y_ex)
                form_loss = self.form_criterion(form_logits, y_form)
                loss = ex_loss + form_loss

                total_loss += loss.item()
                ex_correct += (ex_logits.argmax(dim=1) == y_ex).sum().item()
                form_correct += (form_logits.argmax(dim=1) == y_form).sum().item()
                total += y_ex.size(0)

        return total_loss / len(loader), ex_correct / total, form_correct / total

    def train(self, epochs=50):
        """Full training loop."""
        print(f"\n=== Training Started ===")
        print(f"Epochs: {epochs}")

        best_val_acc = 0
        model_path = Path("models/trained/form_model.pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            train_loss, train_ex_acc, train_form_acc = self.train_epoch()
            val_loss, val_ex_acc, val_form_acc = self.validate(self.val_loader)

            self.scheduler.step()

            # Combined accuracy (average of both tasks)
            val_combined = (val_ex_acc + val_form_acc) / 2

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Ex: {train_ex_acc*100:.1f}% Form: {train_form_acc*100:.1f}% | "
                  f"Val Ex: {val_ex_acc*100:.1f}% Form: {val_form_acc*100:.1f}%")

            if val_combined > best_val_acc:
                best_val_acc = val_combined
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'metadata': self.metadata
                }, model_path)
                print(f"  -> Model saved! Combined: {val_combined*100:.1f}%")

        print(f"\n=== Training Complete ===")
        print(f"Best validation accuracy: {best_val_acc*100:.1f}%")

        # Save metadata for inference
        metadata_path = Path("models/trained/form_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        return best_val_acc

    def evaluate(self):
        """Evaluate on test set."""
        # Load best model
        checkpoint = torch.load("models/trained/form_model.pth", weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_ex_acc, test_form_acc = self.validate(self.test_loader)

        print(f"\n=== Test Results ===")
        print(f"Exercise Accuracy: {test_ex_acc*100:.2f}%")
        print(f"Form Quality Accuracy: {test_form_acc*100:.2f}%")
        print(f"Combined: {(test_ex_acc + test_form_acc)/2*100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    trainer = MultiTaskTrainer(
        data_dir="data/processed_form",
        batch_size=args.batch_size,
        lr=args.lr
    )

    trainer.train(epochs=args.epochs)
    trainer.evaluate()


if __name__ == "__main__":
    main()
