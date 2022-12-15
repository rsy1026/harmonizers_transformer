import numpy as np
import math
import os
import gc
import sys 
sys.path.append("./utils")
sys.path.append("./models")
import time
from glob import glob
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Manager

from models import STHarm, VTHarm
from models.STHarm import Mask, Compress
from process_data import FeatureIndex
from utils.parse_utils import *   
from data_loader import *


## TRAINER ##
def main(dataset=None,
         exp_name=None,
         checkpoint_num=0,
         device_num=2,
         batch_size=16, # 128
         batch_size_val=4, # 128 
         total_epoch=100,
         hidden=256,
         n_layers=4):

    start_time = time.time()
    
    # LOAD DATA
    datapath = './'
    # result_path = os.path.join(datapath, 'results')
    model_path = os.path.join(datapath, 'trained')
    train_data = os.path.join(datapath, '{}_train.h5'.format(dataset))
    val_data = os.path.join(datapath, '{}_val.h5'.format(dataset))
    FI = FeatureIndex(dataset=dataset)

    # LOAD DATA LOADER 
    if dataset == "CMD":
        CustomDataset = CMDDataset
        PadCollate = CMDPadCollate

    elif dataset == "HLSD":
        CustomDataset = HLSDDataset
        PadCollate = HLSDPadCollate

    with h5py.File(train_data, "r") as f:
        train_x = np.asarray(f["x"])
        train_k = np.asarray(f["c"])
        train_n = np.asarray(f["n"])
        train_m = np.asarray(f["m"])
        train_y = np.asarray(f["y"])
        
    with h5py.File(val_data, "r") as f:
        val_x = np.asarray(f["x"])
        val_k = np.asarray(f["c"])
        val_n = np.asarray(f["n"])
        val_m = np.asarray(f["m"])
        val_y = np.asarray(f["y"])

    train_len = len(train_x)
    val_len = len(val_x)
    step_size = int(np.ceil(train_len / batch_size))

    _time0 = time.time()
    load_data_time = np.round(_time0 - start_time, 3)
    print("LOADED DATA")
    print("__time spent for loading data: {} sec".format(load_data_time))

    # LOAD DEVICE
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")

    # LOAD MODEL
    if exp_name == "STHarm":
        MODEL = STHarm.Harmonizer
    elif exp_name == "VTHarm" or exp_name == "rVTHarm":
        MODEL = VTHarm.Harmonizer

    model = MODEL(hidden=hidden, n_layers=n_layers, device=device)
    model.to(device)
    trainer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=trainer, lr_lambda=lambda epoch: 0.95 ** epoch)

    # model_path_ = "./trained/{}".format(exp_num)
    # checkpoint = torch.load(model_path_)
    # model.load_state_dict(checkpoint['state_dict'])
    # trainer.load_state_dict(checkpoint['optimizer'])

    _time1 = time.time()
    load_graph_time = np.round(_time1 - _time0, 3)
    print("LOADED GRAPH")
    print("__time spent for loading graph: {} sec".format(load_graph_time))
    print() 
    print("Start training...")
    print("** step size: {}".format(step_size))
    print()
    bar = 'until next 20th steps: '
    rest = '                    |' 
    shuf = 0
 
    # TRAIN
    start_train_time = time.time()
    prev_epoch_time = start_train_time
    model.train()

    loss_list = list()
    val_loss_list = list()
    # loss_list = checkpoint['loss'].tolist()
    # val_loss_list = checkpoint['val_loss'].tolist()

    # load data loader
    train_dataset = CustomDataset(
        train_x, train_k, train_n, train_m, train_y, device=device)
    val_dataset = CustomDataset(
        val_x, val_k, val_n, val_m, val_y, device=device)

    generator = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, 
        collate_fn=PadCollate(), shuffle=True, drop_last=True, pin_memory=False)
    generator_val = DataLoader(
        val_dataset, batch_size=batch_size_val, num_workers=0, 
        collate_fn=PadCollate(), shuffle=False, pin_memory=False)

    for epoch in range(checkpoint_num, total_epoch):

        epoch += 1
        model.train()
        
        for step, sample in enumerate(generator):

            # load batch
            x, k, n, m, y, clab = sample
            # x, k, n, m, y, clab = next(iter(generator))
            x = x.long().to(device)
            k = k.float().to(device)
            n = n.float().to(device)
            m = m.float().to(device)
            y = y.long().to(device)
            clab = clab.float().to(device)

            step += 1         

            ## GENERATOR ## 
            trainer.zero_grad()

            if exp_name == "STHarm":
                # forward
                chord, kq_attn = model(x, n, m, y) 

                # compute loss 
                mask = Mask()
                loss = ST_loss_fn(chord, y, m, mask) 

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                trainer.step()
                loss_list.append([loss.detach().item()])

                if step % 1 == 0:
                    bar += '='
                    rest = rest[1:]
                    print(bar+'>'+rest, end='\r')

                if step % 20 == 0:
                    # print losses 
                    print()
                    print("[{} --> epoch: {} / step: {}]\n".format(exp_name, epoch, step) + \
                    "   --GENERATOR LOSS--\n" + \
                    "           recon_chord: {:06.4f}\n".format(loss))
                    print()
                    bar = 'until next 20th steps: '
                    rest = '                    |'

            elif exp_name == "VTHarm":
                # forward
                c_moments, c, chord, kq_attn = model(x, k, n, m, y) 

                # compute loss 
                mask = Mask()
                loss, recon_chord, kld_c = VT_loss_fn(
                    c_moments, c, chord, y, m, clab, mask) 

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                trainer.step()
                loss_list.append(
                    [loss.detach().item(), \
                    recon_chord.detach().item(), kld_c.detach().item()])

                if step % 1 == 0:
                    bar += '='
                    rest = rest[1:]
                    print(bar+'>'+rest, end='\r')

                if step % 20 == 0:
                    # print losses 
                    print()
                    print("[{} --> epoch: {} / step: {}]\n".format(exp_name, epoch, step) + \
                    "   --GENERATOR LOSS--\n" + \
                    "           recon_chord: {:06.4f} / kld_c: {:06.4f}\n".format(-recon_chord, kld_c))
                    print()
                    bar = 'until next 20th steps: '
                    rest = '                    |'

            elif exp_name == "rVTHarm":
                # forward
                c_moments, c, chord, kq_attn = model(x, k, n, m, y) 

                # compute loss 
                mask = Mask()
                loss, recon_chord, kld_c, reg_loss = rVT_loss_fn(
                    c_moments, c, chord, y, m, clab, mask) 

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                trainer.step()
                loss_list.append(
                    [loss.detach().item(), \
                    recon_chord.detach().item(), kld_c.detach().item(), reg_loss.detach().item()])

                if step % 1 == 0:
                    bar += '='
                    rest = rest[1:]
                    print(bar+'>'+rest, end='\r')

                if step % 20 == 0:
                    # print losses 
                    print()
                    print("[{} --> epoch: {} / step: {}]\n".format(exp_name, epoch, step) + \
                    "   --GENERATOR LOSS--\n" + \
                    "           recon_chord: {:06.4f} / kld_c: {:06.4f}\n".format(-recon_chord, kld_c) + \
                    "           reg_loss: {:06.4f}\n".format(-reg_loss))
                    print()
                    bar = 'until next 20th steps: '
                    rest = '                    |'

            gc.collect()
 
        _time2 = time.time()
        epoch_time = np.round(_time2 - prev_epoch_time, 3)
        print()
        print()
        print("------------------EPOCH {} finished------------------".format(epoch))
        print()
        print("==> time spent for this epoch: {} sec".format(epoch_time))
        print("==> loss: {:06.4f}".format(loss))  

        model.eval()

        Xv, Kv, Nv, Mv, Yv, CLABv = next(iter(generator_val))
        Xv = Xv.long().to(device)
        Kv = Kv.float().to(device)
        Nv = Nv.float().to(device)
        Mv = Mv.float().to(device)
        Yv = Yv.long().to(device)
        CLABv = CLABv.float().to(device)

        if exp_name == "STHarm":
            # forward
            chord, kq_attn = model(Xv, Nv, Mv, Yv) 

            # generate
            chord_ = model.test(Xv, Nv, Mv)

            # compute loss 
            mask = Mask()
            val_loss = ST_loss_fn(chord, Yv, Mv, mask) 

            val_loss_list.append([val_loss.detach().item()])

            # print losses
            print()
            print()
            print("==> [{} --> epoch {}] Validation:\n".format(exp_name, epoch) + \
            "   --GENERATOR LOSS--\n" + \
            "           recon_chord: {:06.4f}\n".format(val_loss))

        elif exp_name == "VTHarm":
            # forward
            c_moments, c, chord, kq_attn = model(Xv, Kv, Nv, Mv, Yv) 

            # generate
            chord_, kq_attn_ = model.test(Xv, Kv, Nv, Mv)

            # compute loss 
            mask = Mask()
            val_loss, recon_chord, kld_c = VT_loss_fn(
                c_moments, c, chord, Yv, Mv, CLABv, mask) 

            val_loss_list.append(
                [val_loss.detach().item(),
                recon_chord.detach().item(), kld_c.detach().item()])

            # print losses
            print()
            print()
            print("==> [{} --> epoch {}] Validation:\n".format(exp_name, epoch) + \
            "   --GENERATOR LOSS--\n" + \
            "           recon_chord: {:06.4f} / kld_c: {:06.4f}\n".format(-recon_chord, kld_c))

        elif exp_name == "rVTHarm":
            # forward
            c_moments, c, chord, kq_attn = model(Xv, Kv, Nv, Mv, Yv) 

            # generate
            chord_, kq_attn_ = model.test(Xv, Kv, Nv, Mv)

            # compute loss 
            mask = Mask()
            val_loss, recon_chord, kld_c, reg_loss = rVT_loss_fn(
                c_moments, c, chord, Yv, Mv, CLABv, mask) 

            val_loss_list.append(
                [val_loss.detach().item(),
                recon_chord.detach().item(), kld_c.detach().item(), reg_loss.detach().item()])

            # print losses
            print()
            print()
            print("==> [{} --> epoch {}] Validation:\n".format(exp_name, epoch) + \
            "   --GENERATOR LOSS--\n" + \
            "           recon_chord: {:06.4f} / kld_c: {:06.4f}\n".format(-recon_chord, kld_c) + \
            "           reg_loss: {:06.4f}\n".format(-reg_loss))
        
        print()
        bar = 'until next 20th steps: '
        rest = '                    |' 
        print()
        print("------------------EPOCH {} finished------------------".format(epoch))
        print()

        scheduler.step()

        # save checkpoint & loss
        if epoch % 4 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': trainer.state_dict(),
                'loss_train': np.asarray(loss_list),
                'loss_val': np.asarray(val_loss_list)},
                os.path.join(model_path, "{}_{}".format(exp_name, dataset)))

            _time3 = time.time()
            end_train_time = np.round(_time3 - start_train_time, 3)  
            print("__time spent for entire training: {} sec".format(end_train_time))

        prev_epoch_time = time.time()     
        shuf += 1


# LOSS FUNCTIONS
def ST_loss_fn(chord, y, m, mask):

    # VAE losses
    n, t = y.size(0), y.size(1)

    recon_chord = torch.mean(mask(F.cross_entropy(
        chord.view(-1, 73), y.view(-1), 
        reduction='none').view(n, t, 1), m.transpose(1, 2)))

    return recon_chord

def VT_loss_fn(c_moments, c, chord, y, m, clab, mask):

    # VAE losses
    n, t = y.size(0), y.size(1)

    recon_chord = -torch.mean(mask(F.cross_entropy(
        chord.view(-1, 73), y.view(-1), 
        reduction='none').view(n, t, 1), m.transpose(1, 2)))

    kld_c = torch.mean(kld(*c_moments))

    # VAE ELBO
    elbo = recon_chord - 0.1*kld_c

    # total loss
    total_loss = -elbo # negative to minimize

    return total_loss, recon_chord, kld_c

def rVT_loss_fn(c_moments, c, chord, y, m, clab, mask):

    # VAE losses
    n, t = y.size(0), y.size(1)

    recon_chord = -torch.mean(mask(F.cross_entropy(
        chord.view(-1, 73), y.view(-1), 
        reduction='none').view(n, t, 1), m.transpose(1, 2)))

    kld_c = torch.mean(kld(*c_moments))

    # regression loss
    M = c.size(0)
    ss = c[:,0]
    ss_l = clab
    ss1 = ss.unsqueeze(0).expand(M, M)
    ss2 = ss.unsqueeze(1).expand(M, M)
    ss_l1 = ss_l.unsqueeze(0).expand(M, M)
    ss_l2 = ss_l.unsqueeze(1).expand(M, M)
    ss_D = ss1 - ss2 
    ss_l_D = ss_l1 - ss_l2 
    reg_dist = (torch.tanh(ss_D) - torch.sign(ss_l_D))**2
    reg_loss = -torch.mean(reg_dist)

    # VAE ELBO
    elbo = recon_chord - 0.1*kld_c + reg_loss

    # total loss
    total_loss = -elbo # negative to minimize

    return total_loss, recon_chord, kld_c, reg_loss

def kld(mu, logvar, q_mu=None, q_logvar=None):
    '''
    KL(N(mu, var)||N(qmu, qvar))
        --> -0.5 * (1 + logvar - q_logvar 
            - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar)) 
    '''
    if q_mu is None:
        q_mu = torch.zeros_like(mu)
    if q_logvar is None:
        q_logvar = torch.zeros_like(logvar)

    return -0.5 * (1 + logvar - q_logvar - \
        (torch.pow(mu - q_mu, 2) + torch.exp(logvar)) / torch.exp(q_logvar))





if __name__ == "__main__":
    '''
    python3 trainer.py [dataset] [exp_name]

    - dataset: CMD / HLSD 
    - exp_name: STHarm / VTHarm / rVTHarm
    '''
    dataset = sys.argv[1]
    exp_name = sys.argv[2]
    main(dataset=dataset, exp_name=exp_name)



