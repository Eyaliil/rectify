"""
Model Training Script for Exercise Classification

This script trains the SensorClassifier model on FlexTail sensor data.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from models.sensor_classifier import SensorClassifier, LightweightSensorClassifier
from prepare_dataset import ExerciseDatasetBuilder


class SensorDataset(Dataset):
    """PyTorch Dataset for sensor data."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ExerciseClassifierTrainer:
    """Trainer class for exercise classification model."""

    def __init__(
        self,
        model_type='lstm',  # 'lstm' or 'cnn'
        batch_size=32,
        learning_rate=1e-3,
        num_epochs=50,
        device=None
    ):
        self.model_type = model_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        print(f"Using device: {self.device}")

    def load_data(self, data_dir="data/processed"):
        """Load preprocessed dataset."""
        builder = ExerciseDatasetBuilder()
        dataset = builder.load_dataset(data_dir)

        self.metadata = {
            'label2id': dataset['label2id'],
            'id2label': dataset['id2label'],
            'features': dataset['features'],
            'window_size': dataset['window_size'],
            'num_classes': len(dataset['label2id']),
            'input_size': len(dataset['features'])
        }

        print(f"\n=== Dataset Loaded ===")
        print(f"Classes: {list(self.metadata['label2id'].keys())}")
        print(f"Features: {self.metadata['features']}")
        print(f"Input shape: ({self.metadata['window_size']}, {self.metadata['input_size']})")

        # Create datasets
        self.train_dataset = SensorDataset(dataset['X_train'], dataset['y_train'])
        self.val_dataset = SensorDataset(dataset['X_val'], dataset['y_val'])
        self.test_dataset = SensorDataset(dataset['X_test'], dataset['y_test'])

        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")

    def create_model(self):
        """Initialize the model."""
        if self.model_type == 'lstm':
            self.model = SensorClassifier(
                input_size=self.metadata['input_size'],
                hidden_size=128,
                num_layers=2,
                num_classes=self.metadata['num_classes'],
                dropout=0.3
            )
        elif self.model_type == 'cnn':
            self.model = LightweightSensorClassifier(
                input_size=self.metadata['input_size'],
                num_classes=self.metadata['num_classes'],
                dropout=0.3
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = self.model.to(self.device)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel: {self.model_type.upper()}")
        print(f"Parameters: {num_params:,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self):
        """Full training loop."""
        print(f"\n=== Training Started ===")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")

        best_val_acc = 0
        best_epoch = 0

        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print progress
            print(f"Epoch {epoch+1}/{self.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self.save_model('best_model.pth')

            # Step scheduler
            self.scheduler.step()

        print(f"\n=== Training Complete ===")
        print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")

    def evaluate(self):
        """Evaluate on test set."""
        print("\n=== Evaluating on Test Set ===")

        # Load best model
        self.load_model('best_model.pth')

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Accuracy
        accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Classification report
        print("\n" + classification_report(
            all_labels,
            all_preds,
            target_names=list(self.metadata['label2id'].keys())
        ))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.metadata['label2id'].keys()),
            yticklabels=list(self.metadata['label2id'].keys())
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved to confusion_matrix.png")

    def plot_training_curves(self):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("Training curves saved to training_curves.png")

    def save_model(self, filename='model.pth'):
        """Save model and metadata."""
        output_dir = Path('models/trained')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'metadata': self.metadata
        }, output_dir / filename)

        # Save metadata separately
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Model saved to {output_dir / filename}")

    def load_model(self, filename='model.pth'):
        """Load trained model."""
        checkpoint = torch.load(Path('models/trained') / filename, map_location=self.device)

        self.model_type = checkpoint['model_type']
        self.metadata = checkpoint['metadata']

        # Create model
        self.create_model()

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

        print(f"Model loaded from {filename}")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train exercise classification model')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'cnn'],
                        help='Model architecture (default: lstm)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Processed data directory')

    args = parser.parse_args()

    # Create trainer
    trainer = ExerciseClassifierTrainer(
        model_type=args.model,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs
    )

    # Load data
    trainer.load_data(args.data_dir)

    # Create model
    trainer.create_model()

    # Train
    trainer.train()

    # Evaluate
    trainer.evaluate()

    # Plot results
    trainer.plot_training_curves()

    print("\n=== Training Pipeline Complete ===")
    print("Model ready for deployment!")


if __name__ == "__main__":
    main()
