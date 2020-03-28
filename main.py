# -*- encoding: utf-8 -*-
#'''
#@file_name    :main.py
#@description    :
#@time    :2020/02/12 13:46:28
#@author    :Cindy, xd Zhang 
#@version   :0.1
#'''
import argparse
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_loader import *
import time
import os
import network 
from optimiser import ScheduledOptim
USE_CUDA = torch.cuda.is_available()  
def str2bool(v):
    """ str2bool """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
def arg_config():
    """ config """
    project_root_dir=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    parser = argparse.ArgumentParser()
    # Network CMD参数组
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument('-e',"--embedding_size", type=int, default=64)
    net_arg.add_argument('-hi',"--d_hidden", type=int, default=256)
    net_arg.add_argument("--n_layers", type=int, default=6)
    net_arg.add_argument("--dropout", type=float, default=0.1)
    net_arg.add_argument("--d_k", type=int, default=64)
    net_arg.add_argument("--d_v", type=int, default=64)
    net_arg.add_argument("--n_head", type=int, default=8)
    # Training / Testing CMD参数组
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--n_warmup_steps", type=int, default=4000)
    train_arg.add_argument('-bs',"--batch_size", type=int, default=20)
    train_arg.add_argument('-r',"--run_type", type=str, default="train",
     choices=['train', 'test'])
    train_arg.add_argument('-lr',"--lr", type=float, default=0.1)
    train_arg.add_argument("--end_epoch", type=int, default=80)
    gen_arg = parser.add_argument_group("Generation")
    # MISC ：logs,dirs and gpu config
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument('-u', "--use_gpu", type=str2bool, default=True)
    misc_arg.add_argument('-p',"--log_steps", type=int, default=10)
    misc_arg.add_argument('-s',"--save_iteration", type=int, default=40,help='Every save_iteration iteration(s) save checkpoint model ')   
    #路径参数
    misc_arg.add_argument('-con',"--continue_training", type=str, default=" ")
    config = parser.parse_args()
    return config
def main():
    config=arg_config()
    print("-Loading dataset ...")
    DataSet=My_dataset(config.run_type)
    train_loader = DataLoader(dataset=DataSet,\
         shuffle=False, batch_size=config.batch_size,drop_last=True,collate_fn=collate_fn)
    print("-building model ...")
    crit=nn.CrossEntropyLoss(ignore_index=PAD_token)
    model=network.Transformer(config,DataSet.voc.n_words,crit)
    if  config.use_gpu and USE_CUDA :
        print('**work with solo-GPU **')
        network.Global_device = torch.device("cuda:0" )
        model.to(network.Global_device)
    optimizer = ScheduledOptim(
            optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            config.lr, config.embedding_size, config.n_warmup_steps)
    model.train(train_loader,optimizer)

if __name__ == "__main__":
    #配置解析CMD 参数
    main()