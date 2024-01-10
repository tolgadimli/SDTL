import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from DistOptimizers import CommOptimizer

from options import Options
from utils import *
import os


def train(model, device, train_loader, optimizer, scheduler, epoch):
    train_loss = train_accuracy = 0
    train_begin_time = time.time()
    model.train()
    try:
        train_loader.sampler.set_epoch(epoch)
    except:
        pass
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        train_accuracy += pred.eq(target.view_as(pred)).sum().item()
        train_loss += loss.item()

    scheduler.step()
    train_loss /= len(train_loader.sampler)
    train_error = 100.0 * ( 1 - train_accuracy / len(train_loader.sampler) )
    epoch_duration = time.time() - train_begin_time
    
    return epoch_duration, train_loss, train_error

def test(model, device, test_loader):
    center_model = copy.deepcopy(model)
    sync_model_params_and_buffers(center_model)
    test_loss = test_accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = center_model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_accuracy += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_error = 100.0 * ( 1 - test_accuracy / len(test_loader.dataset) )
    return test_loss, test_error

def dist_run(rank, args, accm_workers, world_size, init_method):

    torch.manual_seed(args.seed)
    dist.init_process_group(rank=accm_workers+rank, world_size=world_size, backend='nccl', init_method=init_method)

    cuda_kwargs  = {'num_workers': args.node_gpus}
    train_kwargs = {'batch_size': args.bs}
    test_kwargs  = {'batch_size': args.bs}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    
    train_dataset, test_dataset = get_dataset(args.dataset)
    train_loader = to_dist_train_loader(torch.utils.data.DataLoader(train_dataset, **train_kwargs), is_dist_sampler=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    device = torch.device('cuda:%d'%rank)
    base_model = get_model(args.model, args.dataset).to(device)
    for model_param in base_model.parameters():
        dist.all_reduce_multigpu([model_param.data], op=torch.distributed.ReduceOp.AVG, async_op=False)
    
    print(args.dist_optim_name)
    if not args.dist_optim_name == 'DataParallel':
        model     = copy.deepcopy(base_model).to(device)
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.mom, weight_decay=args.wd, nesterov=True)
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)
        optimizer = CommOptimizer(optimizer, dist_optim_name=args.dist_optim_name, world_size = world_size,
                                        comm_period=args.comm_period, dist_pulling_strength=args.c, local_pulling_strength=args.p)
    else:
        model     = nn.parallel.DistributedDataParallel(copy.deepcopy(base_model).to(device))
        optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.mom, weight_decay=args.wd, nesterov=True)
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.scheduler_gamma)
        
    # train model
    message_file = open(args.message_dir, "w")
    message_file.write("epoch,rank,train_time,train_loss,train_error,test_loss,test_error,lr")
    message_file.close()
   
    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        epoch_time, train_loss, train_error = train(model, device, train_loader, optimizer, scheduler, epoch)
        test_loss, test_error = test(model, device, test_loader)
        cur_lr = scheduler.get_last_lr()[-1]
        total_train_time += epoch_time 

        message_file = open(args.message_dir, "a")
        _text = "\n%d,%d,%.3f,%.4f,%.3f,%.4f,%.3f,%.6f"\
        %(epoch, rank, total_train_time, train_loss, train_error, test_loss, test_error, cur_lr) 
        message_file.write(_text)
        message_file.close()

    if args.save_model:
        print("Saving model parameters.")
        state = model.state_dict()
        torch.save(state, os.path.join(args.export_dir, args.experiment_name,'model_params-end-w%d.pt'%rank))

if __name__ == "__main__":

    args = Options().parse()

    cur_workers   = min(args.node_gpus, torch.cuda.device_count())
    total_workers = torch.zeros(args.num_nodes).int()
    total_workers[args.node_rank] = cur_workers
    if args.num_nodes > 1:
        init_method = "tcp://{ip}:{port}0".format(ip=args.ip_address, port=2432)
        dist.init_process_group(rank=args.node_rank, world_size=args.num_nodes, backend='gloo', init_method=init_method)
        dist.all_reduce(total_workers, op=torch.distributed.ReduceOp.SUM, async_op=False)
    accm_workers = total_workers[:args.node_rank].sum().item()
    world_size   = total_workers.sum().item()

    init_method = "tcp://{ip}:{port}5".format(ip=args.ip_address, port=2432)
    mp.spawn(fn=dist_run, args=(args, accm_workers, world_size, init_method), nprocs=cur_workers)