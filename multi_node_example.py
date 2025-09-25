#!/usr/bin/env python3
"""
Example demonstrating multi-node distributed training.
This example shows how to run training across multiple machines/nodes.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from faeyon.recipes import ClassifyRecipe, FaeOptimizer
from faeyon.callbacks import EarlyStopping
from faeyon.distributed import get_node_info, is_master_node, is_master_process, barrier


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


def print_node_info():
    """Print information about the current node and process"""
    node_info = get_node_info()
    print(f"Node Rank: {node_info['node_rank']}")
    print(f"Local Rank: {node_info['local_rank']}")
    print(f"Local World Size: {node_info['local_world_size']}")
    print(f"Global Rank: {node_info['rank']}")
    print(f"World Size: {node_info['world_size']}")
    print(f"Is Master Node: {is_master_node()}")
    print(f"Is Master Process: {is_master_process()}")


def train_multi_node():
    """Example: Multi-node distributed training"""
    print("=== Multi-Node Distributed Training ===")
    
    # Print node information
    if is_master_process():
        print("Node Information:")
        print_node_info()
        print()
    
    # Create data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model and optimizer
    model = create_model()
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create callbacks (only on master process)
    callbacks = []
    if is_master_process():
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5)
        ]
    
    # Create recipe with distributed training
    recipe = ClassifyRecipe(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        callbacks=callbacks,
        use_distributed=True
    )
    
    # Print training setup
    if is_master_process():
        print(f"Training on {recipe.world_size} processes across multiple nodes")
        print(f"Device: {recipe.device}")
        print(f"Node Rank: {recipe.node_rank}")
        print()
    
    # Synchronize all processes before training
    barrier()
    
    # Train
    history = recipe.train(
        train_loader,
        val_data=val_loader, 
        min_period="30s", 
        max_period="2m", 
        val_freq=10, 
    )
    
    # Synchronize after training
    barrier()
    
    # Print results (only on master process)
    if is_master_process():
        print(f"Training completed: {history['total_steps']} steps in {history['total_time']:.2f}s")
        print(f"Min period reached: {history['min_period_reached']}")
        print(f"Max period reached: {history['max_period_reached']}")
        print(f"Early stopped: {history['early_stopped']}")
    
    # Cleanup
    recipe.cleanup()


def main():
    """Main function for multi-node training"""
    print("=== Faeyon Multi-Node Training Example ===")
    print("This example demonstrates distributed training across multiple nodes.")
    print()
    
    # Check if we're in a distributed environment
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print("This example is designed to run with torchrun for multi-node training.")
        print("To run on multiple nodes, use:")
        print()
        print("# On node 0 (master):")
        print("torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 --master_port=29500 multi_node_example.py")
        print()
        print("# On node 1:")
        print("torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 --master_port=29500 multi_node_example.py")
        print()
        print("Replace 192.168.1.100 with the actual IP address of the master node.")
        print()
        print("For single-node multi-GPU testing:")
        print("torchrun --nproc_per_node=2 multi_node_example.py")
        return
    
    try:
        train_multi_node()
    except Exception as e:
        print(f"Multi-node training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
