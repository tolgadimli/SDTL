import torch
import torch.distributed as dist
import torch.utils.data as udata

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models import *

def sync_model_params_and_buffers(model):
    ''' synchronize model params and buffers '''
    current_index = 0
    for parameter in model.parameters():
        numel = parameter.data.numel()
        dist.all_reduce_multigpu([parameter.data], op=torch.distributed.ReduceOp.AVG, async_op=False)
        current_index += numel

    current_index = 0
    for parameter in model.buffers():
        numel = parameter.data.numel()
        dist.all_reduce_multigpu([parameter.data], op=torch.distributed.ReduceOp.AVG, async_op=False)
        current_index += numel

def to_dist_train_loader(train_loader, is_dist_sampler=False):
    ''' convert local train data loader to distributed train data loader by modifying the arguments '''
    if is_dist_sampler:
        train_sampler = udata.distributed.DistributedSampler(train_loader.dataset, shuffle=True, drop_last=train_loader.drop_last)
        return udata.DataLoader(dataset=train_loader.dataset,
                                batch_size=train_loader.batch_size,
                                num_workers=train_loader.num_workers,
                                pin_memory=True, persistent_workers=True, sampler = train_sampler)
    else:
        return udata.DataLoader(dataset=train_loader.dataset,
                                drop_last=train_loader.drop_last,
                                batch_size=train_loader.batch_size,
                                num_workers=train_loader.num_workers,
                                pin_memory=True, persistent_workers=True, shuffle=True)



def get_num_classes(dataset_name):

    if dataset_name.lower() == 'cifar10' or dataset_name.lower() == 'fashionmnist':
        return 10
    elif dataset_name.lower() == 'cifar100':
        return 100
    elif dataset_name.lower() == 'iamgenet':
        return 1000
    else:
        raise ValueError('Invalid dataset name!')

def get_dataset(dataset_name):

    if dataset_name.lower() == 'fashionmnist':
        train_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_transform=train_transform
        Dataset = datasets.FashionMNIST

    elif 'cifar' in dataset_name.lower():
        
        if dataset_name.lower() == 'cifar10':
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))                       
            Dataset = datasets.CIFAR10

        elif dataset_name.lower() == 'cifar100':
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))                          
            Dataset = datasets.CIFAR100

        train_transform = transforms.Compose([ transforms.RandomHorizontalFlip(),
                                                transforms.RandomCrop(size= 32, padding=4),
                                                transforms.ToTensor(), normalize ])
        test_transform = transforms.Compose([  transforms.ToTensor(), normalize])
                        
    else:
        raise ValueError('Invalid dataset name!')

    # Downloading the dataset and getting train/test portions.
    dataset_download = dist.get_rank() % torch.cuda.device_count() == 0
    if dataset_download:
        train_data = Dataset('./data', train=True, download=dataset_download)
        test_data = Dataset('./data', train=False, download=dataset_download)
    dist.barrier()
    train_data = Dataset('./data', train=True, download=False, transform=train_transform)
    test_data = Dataset('./data', train=False, download=False, transform=test_transform)

    return train_data, test_data



def get_model(model_name, dataset_name):


    num_classes = get_num_classes(dataset_name)

    if model_name.lower() == 'simplenetwork':
        model = NeuralNetwork()

    elif model_name == 'vgg16':  
        return VGG16(num_classes = num_classes) 
    elif model_name == 'resnet20':  
        return ResNet20(num_classes = num_classes) 
    elif model_name == 'resnet56':
        return ResNet56(num_classes=num_classes) 
    elif model_name == 'pyramidnet':
        return PyramidNet(110, 270, num_classes)
    elif model_name == 'densenet':
        return densenet121(num_class=num_classes)
    elif model_name == 'wideresnet':
        print('Using WideResNet')
        return Wide_ResNet(28, 10, num_classes=num_classes)
    elif model_name == 'resnet50':
        return torchvision.models.resnet50(pretrained=False)

    else:
        raise ValueError('Invalid model name!')
    
    return model