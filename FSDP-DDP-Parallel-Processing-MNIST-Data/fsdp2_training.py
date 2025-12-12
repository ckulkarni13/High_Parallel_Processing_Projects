"""
FSDP2 (Fully Sharded Data Parallel) Training Script
Uses loop-based layer wrapping pattern
"""

import argparse
import time
import json
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from common import Net, setup, cleanup, train, test

import warnings
warnings.filterwarnings('ignore')


def fsdp2_main(rank, world_size, args):
    """FSDP2 training function with loop-based layer wrapping"""
    setup(rank, world_size)
    
    if rank == 0:
        print(f"\n[FSDP2] Starting training on {world_size} CPUs")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if rank == 0:
        datasets.MNIST('./data', train=True, download=True, transform=transform)
        datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    dist.barrier()
    
    dataset1 = datasets.MNIST('./data', train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)
    
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cpu_kwargs = {'num_workers': 0, 'pin_memory': False, 'shuffle': False}
    train_kwargs.update(cpu_kwargs)
    test_kwargs.update(cpu_kwargs)
    
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    # FSDP2: Loop-based layer wrapping pattern
    model = Net()
    
    # Define layers to wrap
    layers_to_wrap = ['conv1', 'conv2', 'fc1', 'fc2']
    
    if rank == 0:
        print(f"[FSDP2] Applying loop-based wrapping to {len(layers_to_wrap)} layers...")
    
    # Loop through and wrap each layer (Transformer-style pattern)
    for layer_name in layers_to_wrap:
        layer = getattr(model, layer_name)
        setattr(model, layer_name, fully_shard(layer))
        if rank == 0:
            print(f"[FSDP2]   ✓ {layer_name} wrapped")
    
    # Wrap the entire model
    model = fully_shard(model)
    
    if rank == 0:
        print(f"[FSDP2]   ✓ Full model wrapped")
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    init_start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test_loss, test_acc = test(model, rank, world_size, test_loader)
        scheduler.step()
    
    init_end_time = time.time()
    
    if rank == 0:
        elapsed_time = init_end_time - init_start_time
        print(f"\n{'='*60}")
        print(f"FSDP2 Training Complete")
        print(f"Wrapping Pattern: Loop-based layer wrapping")
        print(f"Layers wrapped: {', '.join(layers_to_wrap)} + full model")
        print(f"Total elapsed time: {elapsed_time:.2f} seconds")
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print(f"{'='*60}\n")
        
        with open('fsdp2_results.json', 'w') as f:
            json.dump({
                'method': 'FSDP2',
                'time': elapsed_time,
                'accuracy': test_acc,
                'world_size': world_size,
                'epochs': args.epochs,
                'wrapping_policy': f'loop-based: {", ".join(layers_to_wrap)} + full model',
                'layers_wrapped': layers_to_wrap
            }, f, indent=2)
    
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSDP2 Training')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--world-size', type=int, default=4)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    
    print(f"Running FSDP2 with {args.world_size} CPUs, {args.epochs} epochs")
    mp.spawn(fsdp2_main, args=(args.world_size, args), nprocs=args.world_size, join=True)
