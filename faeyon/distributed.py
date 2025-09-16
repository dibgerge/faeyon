"""
Distributed training utilities for Faeyon.

This module provides utilities for multi-node and multi-GPU distributed training,
including setup, communication primitives, and synchronization functions.
"""

import os
import time
from typing import Dict, List, Tuple, Optional
import torch


def setup_distributed(backend: str = None, init_method: str = None) -> Tuple[int, int, int, int]:
    """Initialize distributed training for multi-node support
    
    Args:
        backend: Communication backend ('nccl', 'gloo', 'mpi')
        init_method: Initialization method ('env://', 'file://', 'tcp://')
    
    Returns:
        Tuple of (rank, world_size, local_rank, node_rank)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Auto-detect backend if not specified
        if backend is None:
            if torch.cuda.is_available():
                backend = 'nccl'
            else:
                backend = 'gloo'
        
        # Auto-detect init method if not specified
        if init_method is None:
            if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
                init_method = 'env://'
            else:
                init_method = 'env://'
        
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
        
        # Calculate node rank
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        node_rank = rank // local_world_size
        
        return rank, world_size, local_rank, node_rank
    return None, 1, 0, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_node_info() -> Dict[str, int]:
    """Get information about the current node in multi-node setup
    
    Returns:
        Dictionary with node information
    """
    if not torch.distributed.is_initialized():
        return {"node_rank": 0, "local_rank": 0, "local_world_size": 1, "world_size": 1}
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    node_rank = rank // local_world_size
    
    return {
        "node_rank": node_rank,
        "local_rank": local_rank,
        "local_world_size": local_world_size,
        "world_size": world_size,
        "rank": rank
    }


def is_master_node() -> bool:
    """Check if current process is on the master node (node_rank == 0)"""
    if not torch.distributed.is_initialized():
        return True
    
    node_info = get_node_info()
    return node_info["node_rank"] == 0


def is_master_process() -> bool:
    """Check if current process is the master process (rank == 0)"""
    if not torch.distributed.is_initialized():
        return True
    
    return torch.distributed.get_rank() == 0


def barrier() -> None:
    """Synchronize all processes across all nodes"""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def all_reduce_tensor(tensor: torch.Tensor, op: str = 'sum') -> torch.Tensor:
    """All-reduce a tensor across all processes in all nodes
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ('sum', 'mean', 'max', 'min')
    
    Returns:
        Reduced tensor
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    # Convert string to torch.distributed.ReduceOp
    reduce_op_map = {
        'sum': torch.distributed.ReduceOp.SUM,
        'mean': torch.distributed.ReduceOp.SUM,  # Will divide by world_size after
        'max': torch.distributed.ReduceOp.MAX,
        'min': torch.distributed.ReduceOp.MIN
    }
    
    reduce_op = reduce_op_map.get(op, torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(tensor, op=reduce_op)
    
    if op == 'mean':
        tensor = tensor / torch.distributed.get_world_size()
    
    return tensor


def gather_tensors(tensor: torch.Tensor, dst: int = 0) -> List[torch.Tensor]:
    """Gather tensors from all processes to a destination process
    
    Args:
        tensor: Tensor to gather
        dst: Destination rank (default: 0)
    
    Returns:
        List of tensors (only non-empty on destination process)
    """
    if not torch.distributed.is_initialized():
        return [tensor]
    
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    # Create list to hold gathered tensors
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather all tensors
    torch.distributed.all_gather(gathered_tensors, tensor)
    
    if rank == dst:
        return gathered_tensors
    else:
        return []


def setup_distributed_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """Setup DataLoader for distributed training
    
    Args:
        dataset: Dataset to wrap
        batch_size: Batch size per process
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
    
    Returns:
        Distributed DataLoader
    """
    if not torch.distributed.is_initialized():
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
    
    # Create distributed sampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(), shuffle=shuffle
    )
    
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )


def get_available_devices() -> List[torch.device]:
    """Get list of available devices"""
    devices = []
    
    # Add CPU
    devices.append(torch.device('cpu'))
    
    # Add CUDA devices if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f'cuda:{i}'))
    
    return devices


def setup_multi_gpu(model: torch.nn.Module, device_ids: List[int] = None) -> torch.nn.Module:
    """Setup model for multi-GPU training
    
    Args:
        model: Model to wrap
        device_ids: List of GPU device IDs to use
    
    Returns:
        Wrapped model for multi-GPU training
    """
    if not torch.cuda.is_available():
        return model
    
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    if len(device_ids) <= 1:
        return model.to(f'cuda:{device_ids[0]}' if device_ids else 'cuda:0')
    
    # Use DataParallel for single-node multi-GPU
    return torch.nn.DataParallel(model, device_ids=device_ids)


def setup_distributed_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Setup model for distributed training
    
    Args:
        model: Model to wrap
        device: Device to place model on
    
    Returns:
        Wrapped model for distributed training
    """
    if not torch.distributed.is_initialized():
        return model.to(device)
    
    # Move model to device
    model = model.to(device)
    
    # Wrap with DistributedDataParallel
    return torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index] if device.type == 'cuda' else None,
        output_device=device.index if device.type == 'cuda' else None
    )


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from source process to all other processes
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank (default: 0)
    
    Returns:
        Broadcasted tensor
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    torch.distributed.broadcast(tensor, src=src)
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes to all processes
    
    Args:
        tensor: Tensor to gather
    
    Returns:
        List of tensors from all processes
    """
    if not torch.distributed.is_initialized():
        return [tensor]
    
    world_size = torch.distributed.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_tensors, tensor)
    return gathered_tensors


def reduce_scatter_tensor(tensor: torch.Tensor, op: str = 'sum') -> torch.Tensor:
    """Reduce and scatter a tensor across all processes
    
    Args:
        tensor: Tensor to reduce and scatter
        op: Reduction operation ('sum', 'mean', 'max', 'min')
    
    Returns:
        Reduced and scattered tensor
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    # Convert string to torch.distributed.ReduceOp
    reduce_op_map = {
        'sum': torch.distributed.ReduceOp.SUM,
        'mean': torch.distributed.ReduceOp.SUM,
        'max': torch.distributed.ReduceOp.MAX,
        'min': torch.distributed.ReduceOp.MIN
    }
    
    reduce_op = reduce_op_map.get(op, torch.distributed.ReduceOp.SUM)
    torch.distributed.reduce_scatter(tensor, [tensor], op=reduce_op)
    
    if op == 'mean':
        tensor = tensor / torch.distributed.get_world_size()
    
    return tensor


def get_world_size() -> int:
    """Get the total number of processes across all nodes"""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    """Get the rank of the current process"""
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_local_rank() -> int:
    """Get the local rank of the current process within its node"""
    if not torch.distributed.is_initialized():
        return 0
    return int(os.environ.get('LOCAL_RANK', 0))


def get_node_rank() -> int:
    """Get the rank of the current node"""
    if not torch.distributed.is_initialized():
        return 0
    
    rank = torch.distributed.get_rank()
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    return rank // local_world_size


def is_distributed() -> bool:
    """Check if distributed training is initialized"""
    return torch.distributed.is_initialized()


def get_backend() -> Optional[str]:
    """Get the current distributed backend"""
    if not torch.distributed.is_initialized():
        return None
    return torch.distributed.get_backend()
