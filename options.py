import argparse
import os
import datetime

# ---- Usefull Utilities ----
def mkdir(path):
    '''create a single empty directory if it didn't exist
    Parameters: path (str) -- a single directory path'''
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    '''create empty directories if they don't exist
    Parameters: paths (str list) -- a list of directory paths'''
    rmdirs(paths)
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def rmdirs(paths):
    if os.path.exists(paths):
        for file in os.listdir(paths): 
            file_path = os.path.join(paths, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(paths)


class Options():
    ''' This class defines argsions
        adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options
    '''
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        ''' set up arguments '''

        parser.add_argument('--ip_address', type=str, required=True)
        parser.add_argument('--node_rank',  type=int, default = 0) 
        parser.add_argument('--node_gpus',  type=int, default = 4) # number of gpus on the current node, not the total # of nodes!
        parser.add_argument('--num_nodes', type=int, default = 1)

        parser.add_argument('--dist_optim_name', type=str, default = 'DataParallel')
        parser.add_argument('--export_dir', type = str, default='./export')
        parser.add_argument('--experiment_name', type=str, default='experiment')
        parser.add_argument('--message_file', type=str, default='messages.csv')
        parser.add_argument('--datadir', type = str, default='./data', help='data')
        parser.add_argument('--dataset', type = str, default='cifar10', help='dataset name')
        parser.add_argument('--model', type = str, default='resnet20')
        parser.add_argument('--seed', type = int, default=32)


        parser.add_argument('--epochs', type = int, default=20)
        parser.add_argument('--bs', type = int, default=128, help='batch size')
        parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
        parser.add_argument('--wd', type = float, default=1e-4 , help='weight decay')
        parser.add_argument('--mom', type = float, default=0.9, help='momentum')

        parser.add_argument('--scheduler_gamma', type = float, default=0.1, help='lr scheduler decay rate')
        parser.add_argument('--scheduler_step_size', type = int, default=100, help='lr scheduler decay period for steplr') 
        parser.add_argument('--scheduler_milestones', type = str, default='100,150', help='lr drop milestones for multisteplr')


        parser.add_argument('--comm_period', default=4, type=int) 
        parser.add_argument('--c', type = float, default=0.1, help='pulling strength')
        parser.add_argument('--p', type = float, default=0.1, help='proximity pulling strength')
        parser.add_argument('--save_model', default=False, action='store_true')

        # parser.add_argument('--random_sampler', action='store_true')
        # parser.add_argument('--lr_lin', action='store_true')
        # parser.add_argument('--lr_pow', action='store_true')
        # parser.add_argument('--lr_decay', action='store_true')
        # parser.add_argument('--lr_iter_decay', type = int, default=-1)
        # parser.add_argument('--lr_time_decay', type = float, default=-1)
        # parser.add_argument('--lr_decay_coef', type = float, default= 1)
        # parser.add_argument('--alpha', type = float, default=0.9, help='exponential moving averaging')
        # parser.add_argument('--gamma', type = float, default=0.9, help='x to mu pulling strength')
        # parser.add_argument('--etagamma', type = float, default=0.99, help='SGLD step size')
        
        self.initialized = True
        return parser

    def get_argsions(self):
        ''' get argsions from parser '''
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_argsions(self, args):
        ''' Print and save argsions
            It will print both current argsions and default values(if different).
            It will save argsions into a text file / [checkpoints_dir] / args.txt
        ''' 
        message = str(datetime.datetime.now())
        message += '\n----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(args.export_dir, args.experiment_name)
        mkdirs(expr_dir) # first remove existing directory and then create a new one
        file_name = os.path.join(expr_dir, 'args.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')
    
    def parse(self):
        ''' Parse our argsions, create checkpoints directory suffix, and set up gpu device. '''
        args = self.get_argsions()

        args.milestones = [int(el) for el in args.scheduler_milestones.split(',')]
        print(args.milestones)

        args.message_dir = os.path.join(args.export_dir, args.experiment_name, args.message_file)

        if args.dist_optim_name.lower() == "DDP":
            args.dist_optim_name = "DataParallel"

        if args.dataset.lower() == 'fashionmnist':
            args.dataset = 'FashionMNIST'
        elif args.dataset.lower() == 'cifar10':
            args.dataset = 'CIFAR10'
        elif args.dataset.lower() == 'cifar100':
            args.dataset = 'CIFAR100'
        elif args.dataset.lower() == 'imagenet':
            args.dataset = 'ImageNet'
        else:
            raise ValueError("wrong dataset name!")

        assert args.dist_optim_name in ['EASGD', 'LSGD', 'DataParallel'], 'No such distributed optimizer supported'

        self.print_argsions(args)
        self.args = args
        return self.args