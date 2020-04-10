import cv2
import glob
import imageio
import numpy as np
import os
import shutil
import torch
import torchvision.datasets as dset
import re

from torchvision import transforms


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def write_config_to_file(config, save_path):
    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        for arg in vars(config):
            file.write(str(arg) + ': ' + str(getattr(config, arg)) + '\n')


def copy_scripts(dst):
    for file in glob.glob('*.py'):
        shutil.copy(file, dst)

    for d in glob.glob('*/'):
        if '__' not in d and d[0] != '.':
            shutil.copytree(d, os.path.join(dst, d))

def save_ckpt(sagan_obj, model=False, final=False):
    print("Saving ckpt...")

    if final:
        # Save final - both model and state_dict
        torch.save({
                    'step': sagan_obj.step,
                    'G_state_dict': sagan_obj.G.module.state_dict() if hasattr(sagan_obj.G, "module") else sagan_obj.G.state_dict(),    # "module" in case DataParallel is used
                    'G_optimizer_state_dict': sagan_obj.G_optimizer.state_dict(),
                    'D_state_dict': sagan_obj.D.module.state_dict() if hasattr(sagan_obj.D, "module") else sagan_obj.D.state_dict(),    # "module" in case DataParallel is used,
                    'D_optimizer_state_dict': sagan_obj.D_optimizer.state_dict(),
                    }, os.path.join(sagan_obj.config.model_weights_path, '{}_final_state_dict_ckpt_{:07d}.pth'.format(sagan_obj.config.name, sagan_obj.step)))
        torch.save({
                    'step': sagan_obj.step,
                    'G': sagan_obj.G,
                    'G_optimizer': sagan_obj.G_optimizer,
                    'D': sagan_obj.D,
                    'D_optimizer': sagan_obj.D_optimizer,
                    }, os.path.join(sagan_obj.config.model_weights_path, '{}_final_model_ckpt_{:07d}.pth'.format(sagan_obj.config.name, sagan_obj.step)))

    elif model:
        # Save full model (not state_dict)
        torch.save({
                    'step': sagan_obj.step,
                    'G': sagan_obj.G.module if hasattr(sagan_obj.G, "module") else sagan_obj.G,     # "module" in case DataParallel is used
                    'G_optimizer': sagan_obj.G_optimizer,
                    'D': sagan_obj.D.module if hasattr(sagan_obj.D, "module") else sagan_obj.D,     # "module" in case DataParallel is used
                    'D_optimizer': sagan_obj.D_optimizer,
                    }, os.path.join(sagan_obj.config.model_weights_path, '{}_model_ckpt_{:07d}.pth'.format(sagan_obj.config.name, sagan_obj.step)))

    else:
        # Save state_dict
        torch.save({
                    'step': sagan_obj.step,
                    'G_state_dict': sagan_obj.G.module.state_dict() if hasattr(sagan_obj.G, "module") else sagan_obj.G.state_dict(),
                    'G_optimizer_state_dict': sagan_obj.G_optimizer.state_dict(),
                    'D_state_dict': sagan_obj.D.module.state_dict() if hasattr(sagan_obj.D, "module") else sagan_obj.D.state_dict(),
                    'D_optimizer_state_dict': sagan_obj.D_optimizer.state_dict(),
                    'D_3D_state_dict': sagan_obj.D_3D.module.state_dict() if hasattr(sagan_obj.D_3D, "module") else sagan_obj.D_3D.state_dict(),
                    'D_3D_optimizer_state_dict': sagan_obj.D_3D_optimizer.state_dict(),
                    'GRU_state_dict': sagan_obj.gru.module.state_dict() if hasattr(sagan_obj.gru, "module") else sagan_obj.gru.state_dict(),
                    'GRU_optimizer_state_dict': sagan_obj.gru_optimizer.state_dict(),
                    }, os.path.join(sagan_obj.config.model_weights_path, 'ckpt_{:07d}.pth'.format(sagan_obj.step)))


def load_pretrained_model(sagan_obj, test=False):
    # Check for path
    model_save_path = sagan_obj.config.model_weights_path
    mod = glob.glob(model_save_path+'/*.pth')
    resume = True if len(mod) != 0 else False
    if resume:
        if not test:
            print('Resume Training...')
            folder = glob.glob(model_save_path+'/ckpt_*.pth')
            start = extractMax(folder)
            checkpoint = torch.load(model_save_path+'/ckpt_'+str(start).zfill(7)+'.pth')
        else:
            checkpoint = torch.load(model_save_path+'/checkpoint.pth')
        # If we know it is a state_dict (instead of complete model)
        if sagan_obj.config.state_dict_or_model == 'state_dict':
            sagan_obj.start = checkpoint['step'] + 1
            sagan_obj.G.load_state_dict(checkpoint['G_state_dict'])
            sagan_obj.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            if not test:
                sagan_obj.D.load_state_dict(checkpoint['D_state_dict'])
                sagan_obj.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
                sagan_obj.D_3D.load_state_dict(checkpoint['D_3D_state_dict'])
                sagan_obj.D_3D_optimizer.load_state_dict(checkpoint['D_3D_optimizer_state_dict'])
            sagan_obj.gru.load_state_dict(checkpoint['GRU_state_dict'])
            sagan_obj.gru_optimizer.load_state_dict(checkpoint['GRU_optimizer_state_dict'])
        # Else, if we know it is a complete model (and not just state_dict)
        elif sagan_obj.config.state_dict_or_model == 'model':
            sagan_obj.start = checkpoint['step'] + 1
            sagan_obj.G = torch.load(checkpoint['G']).to(sagan_obj.device)
            sagan_obj.G_optimizer = torch.load(checkpoint['G_optimizer'])
            sagan_obj.D = torch.load(checkpoint['D']).to(sagan_obj.device)
            sagan_obj.D_optimizer = torch.load(checkpoint['D_optimizer'])
        # Else try for complete model, then try for state_dict
        else:
            try:
                sagan_obj.start = checkpoint['step'] + 1
                sagan_obj.G.load_state_dict(checkpoint['G_state_dict'])
                sagan_obj.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
                if not test:
                    sagan_obj.D.load_state_dict(checkpoint['D_state_dict'])
                    sagan_obj.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
                    sagan_obj.D_3D.load_state_dict(checkpoint['D_3D_state_dict'])
                    sagan_obj.D_3D_optimizer.load_state_dict(checkpoint['D_3D_optimizer_state_dict'])
                sagan_obj.gru.load_state_dict(checkpoint['GRU_state_dict'])
                sagan_obj.gru_optimizer.load_state_dict(checkpoint['GRU_optimizer_state_dict'])
            except:
                sagan_obj.start = checkpoint['step'] + 1
                sagan_obj.G = torch.load(checkpoint['G']).to(sagan_obj.device)
                sagan_obj.G_optimizer = torch.load(checkpoint['G_optimizer'])
                sagan_obj.D = torch.load(checkpoint['D']).to(sagan_obj.device)
                sagan_obj.D_optimizer = torch.load(checkpoint['D_optimizer'])


def extractMax(input): 
  
    # get a list of all numbers separated by  
    # lower case characters  
    # \d+ is a regular expression which means 
    # one or more digit 
    # output will be like ['100','564','365'] 
    numbers = []
    for i in range(len(input)):
        temp = re.findall('\d+',input[i])
        for j in range(len(temp)):
            temp[j] = int(temp[j])
        numbers.append(max(temp))

    return max(numbers)
def check_for_CUDA(sagan_obj):
    if not sagan_obj.config.disable_cuda and torch.cuda.is_available():
        print("CUDA is available!")
        sagan_obj.device = torch.device('cuda')
        sagan_obj.config.dataloader_args['pin_memory'] = True
    else:
        print("Cuda is NOT available, running on CPU.")
        sagan_obj.device = torch.device('cpu')

    if torch.cuda.is_available() and sagan_obj.config.disable_cuda:
        print("WARNING: You have a CUDA device, so you should probably run without --disable_cuda")
