import torch
from config import Param
from methods.utils import setup_seed
from methods.manager import Manager
def run(args):
    setup_seed(args.seed)
    print("hyper-parameter configurations:")
    print(str(args.__dict__))
    
    manager = Manager(args)
    manager.train(args)


if __name__ == '__main__':
    param = Param() # There are detailed hyper-parameter configurations.
    args = param.args
    print('Use {} optim!'.format(args.optim))