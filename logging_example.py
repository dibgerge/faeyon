#!/usr/bin/env python3
"""
Example demonstrating metrics logging and platform integration.
This example shows how to use TensorBoard, MLFlow, and Weights & Biases logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from faeyon.recipes import ClassifyRecipe, FaeOptimizer
from faeyon.callbacks import EarlyStopping, ModelCheckpoint
from faeyon.loggers import TensorBoardLogger, MLFlowLogger, WandBLogger


def create_sample_data(num_samples=1000, input_dim=784, num_classes=10):
    """Create sample data for demonstration"""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def create_model(input_dim=784, num_classes=10):
    """Create a simple neural network model"""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )


def train_with_tensorboard():
    """Example: Training with TensorBoard logging"""
    print("=== Training with TensorBoard Logging ===")
    
    # Create data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model and optimizer
    model = create_model()
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        TensorBoardLogger(
            log_dir="logs/tensorboard_example",
            metrics=['accuracy', 'precision', 'recall', 'f1'],
            log_freq=5
        )
    ]
    
    # Create recipe
    recipe = ClassifyRecipe(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        callbacks=callbacks
    )
    
    # Train
    history = recipe.train(
        train_loader, 
        min_period="30s", 
        max_period="2m", 
        val_data=val_loader, 
        val_freq=10, 
        verbose=True
    )
    
    print(f"Training completed: {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Min period reached: {history['min_period_reached']}")
    print(f"Max period reached: {history['max_period_reached']}")
    print(f"Early stopped: {history['early_stopped']}")
    print("TensorBoard logs saved to: logs/tensorboard_example")
    print("To view: tensorboard --logdir logs/tensorboard_example")
    
    return history


def train_with_mlflow():
    """Example: Training with MLFlow logging"""
    print("\n=== Training with MLFlow Logging ===")
    
    # Create data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model and optimizer
    model = create_model()
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        MLFlowLogger(
            experiment_name="faeyon_classification",
            metrics=['accuracy', 'precision', 'recall', 'f1'],
            log_freq=5
        )
    ]
    
    # Create recipe
    recipe = ClassifyRecipe(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        callbacks=callbacks
    )
    
    # Train
    history = recipe.train(
        train_loader, 
        min_period="30s", 
        max_period="2m", 
        val_data=val_loader, 
        val_freq=10, 
        verbose=True
    )
    
    print(f"Training completed: {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Min period reached: {history['min_period_reached']}")
    print(f"Max period reached: {history['max_period_reached']}")
    print(f"Early stopped: {history['early_stopped']}")
    print("MLFlow logs saved. To view: mlflow ui")
    
    return history


def train_with_wandb():
    """Example: Training with Weights & Biases logging"""
    print("\n=== Training with Weights & Biases Logging ===")
    
    # Create data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model and optimizer
    model = create_model()
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        WandBLogger(
            project="faeyon-classification",
            metrics=['accuracy', 'precision', 'recall', 'f1'],
            log_freq=5,
            config={
                "learning_rate": 0.001,
                "batch_size": 32,
                "architecture": "MLP"
            }
        )
    ]
    
    # Create recipe
    recipe = ClassifyRecipe(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        callbacks=callbacks
    )
    
    # Train
    history = recipe.train(
        train_loader, 
        min_period="30s", 
        max_period="2m", 
        val_data=val_loader, 
        val_freq=10, 
        verbose=True
    )
    
    print(f"Training completed: {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Min period reached: {history['min_period_reached']}")
    print(f"Max period reached: {history['max_period_reached']}")
    print(f"Early stopped: {history['early_stopped']}")
    print("Weights & Biases logs saved. Check your W&B dashboard.")
    
    return history


def train_with_all_loggers():
    """Example: Training with all logging platforms"""
    print("\n=== Training with All Logging Platforms ===")
    
    # Create data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model and optimizer
    model = create_model()
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create callbacks with all loggers
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(save_path="checkpoints/best_model.pth", monitor='val_loss'),
        TensorBoardLogger(
            log_dir="logs/all_platforms",
            metrics=['accuracy', 'precision', 'recall', 'f1'],
            log_freq=5
        ),
        MLFlowLogger(
            experiment_name="faeyon_comprehensive",
            metrics=['accuracy', 'precision', 'recall', 'f1'],
            log_freq=5
        ),
        WandBLogger(
            project="faeyon-comprehensive",
            metrics=['accuracy', 'precision', 'recall', 'f1'],
            log_freq=5,
            config={
                "learning_rate": 0.001,
                "batch_size": 32,
                "architecture": "MLP",
                "dataset": "synthetic"
            }
        )
    ]
    
    # Create recipe
    recipe = ClassifyRecipe(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        callbacks=callbacks
    )
    
    # Train
    history = recipe.train(
        train_loader, 
        min_period="30s", 
        max_period="2m", 
        val_data=val_loader, 
        val_freq=10, 
        verbose=True
    )
    
    print(f"Training completed: {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Min period reached: {history['min_period_reached']}")
    print(f"Max period reached: {history['max_period_reached']}")
    print(f"Early stopped: {history['early_stopped']}")
    print("\nLogs saved to:")
    print("- TensorBoard: logs/all_platforms")
    print("- MLFlow: Check mlflow ui")
    print("- Weights & Biases: Check your W&B dashboard")
    print("- Model checkpoint: checkpoints/best_model.pth")
    
    return history


def main():
    """Main function demonstrating all logging capabilities"""
    print("=== Faeyon Metrics Logging and Platform Integration Example ===")
    print("This example demonstrates:")
    print("1. Callback stop mechanism (EarlyStopping)")
    print("2. Metrics computation (accuracy, precision, recall, F1)")
    print("3. Platform logging (TensorBoard, MLFlow, Weights & Biases)")
    print("4. Min/Max period training with early stopping")
    
    # Individual platform examples
    try:
        train_with_tensorboard()
    except Exception as e:
        print(f"TensorBoard example failed: {e}")
    
    try:
        train_with_mlflow()
    except Exception as e:
        print(f"MLFlow example failed: {e}")
    
    try:
        train_with_wandb()
    except Exception as e:
        print(f"WandB example failed: {e}")
    
    # Comprehensive example
    try:
        train_with_all_loggers()
    except Exception as e:
        print(f"Comprehensive example failed: {e}")
    
    print("\n=== Installation Requirements ===")
    print("For full functionality, install:")
    print("pip install tensorboard mlflow wandb")
    print("\n=== Usage Notes ===")
    print("1. Early stopping now properly signals the training loop to stop")
    print("2. Metrics are automatically computed and logged")
    print("3. Multiple logging platforms can be used simultaneously")
    print("4. Training respects min/max periods with early stopping")


if __name__ == "__main__":
    main()
