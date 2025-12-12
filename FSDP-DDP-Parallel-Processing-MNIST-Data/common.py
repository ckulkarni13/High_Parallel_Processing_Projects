"""
Common components for MNIST distributed training
Contains: Model, train, test, setup, cleanup functions
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Set environment variables
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'


def setup(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()


class Net(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    """Training function for distributed training"""
    model.train()
    ddp_loss = torch.zeros(2)
    if sampler:
        sampler.set_epoch(epoch)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))


def test(model, rank, world_size, test_loader):
    """Testing function for distributed training"""
    model.eval()
    ddp_loss = torch.zeros(3)
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        test_loss = (ddp_loss[0] / ddp_loss[2]).item()
        accuracy = (100. * ddp_loss[1] / ddp_loss[2]).item()
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1].item()), int(ddp_loss[2].item()), accuracy))
        return test_loss, accuracy
    return 0.0, 0.0
