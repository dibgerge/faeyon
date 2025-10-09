# Multi-Node Distributed Training with Faeyon

This guide explains how to run Faeyon training across multiple nodes (machines) for large-scale distributed training.

## Overview

Multi-node training allows you to scale your training across multiple machines, each potentially having multiple GPUs. This is essential for training large models or processing large datasets that don't fit on a single machine.

## Prerequisites

1. **PyTorch with Distributed Support**: Ensure PyTorch is installed with distributed training support
2. **Network Connectivity**: All nodes must be able to communicate with each other
3. **Shared File System** (optional): For shared data and model checkpoints
4. **Consistent Environment**: Same Python environment and dependencies on all nodes

## Quick Start

### 1. Single Node Multi-GPU (Testing)

```bash
# Test with 2 GPUs on a single node
torchrun --nproc_per_node=2 multi_node_example.py
```

### 2. Multi-Node Training

#### On Master Node (Node 0):
```bash
export MASTER_ADDR="192.168.1.100"  # IP of master node
export MASTER_PORT="29500"
export NNODES=2
export NPROC_PER_NODE=2
export NODE_RANK=0

torchrun \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    multi_node_example.py
```

#### On Worker Node (Node 1):
```bash
export MASTER_ADDR="192.168.1.100"  # IP of master node
export MASTER_PORT="29500"
export NNODES=2
export NPROC_PER_NODE=2
export NODE_RANK=1

torchrun \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    multi_node_example.py
```

### 3. Using the Setup Script

```bash
# Make script executable
chmod +x setup_multi_node.sh

# Run on master node
MASTER_ADDR="192.168.1.100" NODE_RANK=0 ./setup_multi_node.sh multi_node_example.py

# Run on worker node
MASTER_ADDR="192.168.1.100" NODE_RANK=1 ./setup_multi_node.sh multi_node_example.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MASTER_ADDR` | IP address of the master node | `localhost` |
| `MASTER_PORT` | Port for communication | `29500` |
| `NNODES` | Total number of nodes | `2` |
| `NPROC_PER_NODE` | Number of processes per node | `2` |
| `NODE_RANK` | Rank of current node (0 for master) | `0` |
| `LOCAL_RANK` | Local rank within node | `0` |
| `WORLD_SIZE` | Total number of processes | `NNODES * NPROC_PER_NODE` |

### Backend Selection

Faeyon automatically selects the best backend:

- **NCCL**: For CUDA/GPU training (recommended)
- **GLOO**: For CPU training or when NCCL is not available
- **MPI**: For HPC environments (requires MPI installation)

## Code Examples

### Basic Multi-Node Training

```python
from faeyon.recipes import ClassifyRecipe, FaeOptimizer
from faeyon.recipes import is_master_process, get_node_info

# Create model and optimizer
model = create_model()
optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)

# Create recipe with distributed training
recipe = ClassifyRecipe(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    use_distributed=True  # Enable multi-node training
)

# Print node information (only on master process)
if is_master_process():
    node_info = get_node_info()
    print(f"Training on {node_info['world_size']} processes")
    print(f"Node rank: {node_info['node_rank']}")

# Train
history = recipe.train(
    train_loader, 
    min_period="30s", 
    max_period="2m", 
    val_loader, 
    verbose=is_master_process()  # Only master prints
)
```

### Advanced Multi-Node Features

```python
from faeyon.recipes import (
    barrier, all_reduce_tensor, gather_tensors,
    is_master_node, is_master_process
)

# Synchronize all processes
barrier()

# All-reduce a tensor across all processes
loss_tensor = torch.tensor(loss_value)
reduced_loss = all_reduce_tensor(loss_tensor, op='mean')

# Gather tensors from all processes
if is_master_process():
    all_tensors = gather_tensors(some_tensor)
    print(f"Gathered {len(all_tensors)} tensors")

# Check node and process roles
if is_master_node():
    print("This is the master node")
if is_master_process():
    print("This is the master process")
```

## Best Practices

### 1. Data Loading

- Use `DistributedSampler` for proper data distribution
- Ensure each process sees different data
- Consider data sharding for very large datasets

### 2. Logging and Monitoring

- Only log from master process to avoid duplicate output
- Use distributed-aware logging callbacks
- Monitor all nodes for failures

### 3. Checkpointing

- Save checkpoints only from master process
- Use shared storage for model checkpoints
- Implement checkpoint resuming across nodes

### 4. Error Handling

- Implement proper cleanup on failures
- Use try-catch blocks around training loops
- Ensure all processes exit cleanly

### 5. Performance Optimization

- Use appropriate batch sizes per GPU
- Tune learning rate for distributed training
- Monitor communication overhead

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check `MASTER_ADDR` and `MASTER_PORT`
   - Ensure firewall allows communication
   - Verify network connectivity

2. **Hanging on Initialization**
   - Check all nodes are running the same command
   - Verify `NODE_RANK` is unique for each node
   - Ensure all nodes can reach the master

3. **CUDA Out of Memory**
   - Reduce batch size per GPU
   - Use gradient accumulation
   - Check GPU memory usage

4. **Slow Training**
   - Check network bandwidth
   - Optimize data loading
   - Use appropriate backend (NCCL for GPU)

### Debugging

```python
# Print detailed node information
from faeyon.recipes import get_node_info
node_info = get_node_info()
print(f"Node info: {node_info}")

# Check distributed status
print(f"Distributed initialized: {torch.distributed.is_initialized()}")
print(f"Backend: {torch.distributed.get_backend()}")
```

## Cloud Deployment

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: faeyon-multi-node
spec:
  replicas: 2  # Number of nodes
  template:
    spec:
      containers:
      - name: faeyon
        image: faeyon:latest
        command: ["torchrun"]
        args:
        - "--nproc_per_node=2"
        - "--nnodes=2"
        - "--node_rank=$(NODE_RANK)"
        - "--master_addr=$(MASTER_ADDR)"
        - "--master_port=29500"
        - "multi_node_example.py"
        env:
        - name: NODE_RANK
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: MASTER_ADDR
          value: "faeyon-master"
```

### Docker Compose

```yaml
version: '3.8'
services:
  master:
    image: faeyon:latest
    command: >
      torchrun
      --nproc_per_node=2
      --nnodes=2
      --node_rank=0
      --master_addr=master
      --master_port=29500
      multi_node_example.py
    environment:
      - MASTER_ADDR=master
      - MASTER_PORT=29500
      - NODE_RANK=0
  
  worker:
    image: faeyon:latest
    command: >
      torchrun
      --nproc_per_node=2
      --nnodes=2
      --node_rank=1
      --master_addr=master
      --master_port=29500
      multi_node_example.py
    environment:
      - MASTER_ADDR=master
      - MASTER_PORT=29500
      - NODE_RANK=1
    depends_on:
      - master
```

## Performance Considerations

### Scaling Efficiency

- **Linear Scaling**: Ideal case where 2x nodes = 2x speed
- **Communication Overhead**: Network latency and bandwidth
- **Data Loading**: I/O bottlenecks across nodes
- **Model Synchronization**: Gradient averaging overhead

### Optimization Tips

1. **Batch Size**: Increase batch size per GPU for better efficiency
2. **Learning Rate**: Scale learning rate with batch size
3. **Gradient Accumulation**: Use for very large models
4. **Mixed Precision**: Use FP16 for faster training
5. **Data Parallelism**: Ensure good data distribution

## Monitoring

### Metrics to Track

- **Training Speed**: Steps per second across all nodes
- **Communication Time**: Time spent on gradient synchronization
- **Memory Usage**: GPU memory per node
- **Network Utilization**: Bandwidth usage between nodes

### Tools

- **TensorBoard**: Distributed training visualization
- **MLFlow**: Experiment tracking across nodes
- **Weights & Biases**: Real-time monitoring
- **Grafana**: System metrics and performance

## Example Commands

### Single Node Multi-GPU
```bash
torchrun --nproc_per_node=4 multi_node_example.py
```

### Two Nodes, 2 GPUs Each
```bash
# Node 0
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 multi_node_example.py

# Node 1  
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 multi_node_example.py
```

### Four Nodes, 4 GPUs Each
```bash
# Node 0
torchrun --nproc_per_node=4 --nnodes=4 --node_rank=0 --master_addr=192.168.1.100 multi_node_example.py

# Node 1
torchrun --nproc_per_node=4 --nnodes=4 --node_rank=1 --master_addr=192.168.1.100 multi_node_example.py

# Node 2
torchrun --nproc_per_node=4 --nnodes=4 --node_rank=2 --master_addr=192.168.1.100 multi_node_example.py

# Node 3
torchrun --nproc_per_node=4 --nnodes=4 --node_rank=3 --master_addr=192.168.1.100 multi_node_example.py
```

This comprehensive multi-node support enables Faeyon to scale training across any number of machines, making it suitable for the largest models and datasets.
