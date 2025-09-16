#!/bin/bash
# Multi-node training setup script for Faeyon

# Configuration
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
NODE_RANK=${NODE_RANK:-0}

# Script to run
SCRIPT=${1:-"multi_node_example.py"}

echo "=== Faeyon Multi-Node Training Setup ==="
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Number of Nodes: $NNODES"
echo "Processes per Node: $NPROC_PER_NODE"
echo "Current Node Rank: $NODE_RANK"
echo "Script: $SCRIPT"
echo

# Check if torchrun is available
if ! command -v torchrun &> /dev/null; then
    echo "Error: torchrun not found. Please install PyTorch with distributed support."
    exit 1
fi

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: Script '$SCRIPT' not found."
    exit 1
fi

# Run the training
echo "Starting multi-node training..."
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $SCRIPT

echo "Multi-node training completed."
