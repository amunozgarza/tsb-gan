import datetime
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision
import glob
import re
from PIL import Image
from transforms import *
import pandas as pd
import inception_utils
#import GPUtil as GPU
import cv2

from torch.backends import cudnn

import utils
import utils2
from models import Generator, GRU
from dataset import TSNDataSet
import imageio
import shutil

class Tester(object):

    def __init__(self, config):

        # Config
        self.config = config

        self.start = 0 # Unless using pre-trained model

        # Create directories if not exist
        utils.make_folder(self.config.save_path)
        utils.make_folder(self.config.model_weights_path)
        utils.make_folder(self.config.sample_images_path)
        utils.make_folder(self.config.log_path)

        # Copy files
        self.T = 16
        
        if self.config.dataset == "weizmann":
            self.num_of_classes = 10
        if self.config.dataset == "ucf11":
            self.num_of_classes = 11
        if self.config.dataset == "ucf101":
            self.num_of_classes = 101
        if self.config.dataset == "kinetics":
            self.num_of_classes = 600
        if self.config.dataset == "jester":
            self.num_of_classes = 27

        # Build G and D
        self.build_models()


    def create_dataset(self):
        # For fast training
        cudnn.benchmark = True
        #torch.manual_seed(self.config.manual_seed)

        # For BatchNorm
        self.G.eval()

        self.gpu_batches = self.config.batch_size//self.config.batch_size_in_gpu
        labels = np.arange(0, self.num_of_classes)
        for label in labels:
            for gpu_batch in range(self.gpu_batches):

                # Create random noise
                z = self.gen_z()
                class_ = torch.full((self.config.batch_size_in_gpu*self.T,), label).long()
                act = self.gru(z)
                fake_videos = self.G(act, class_)
                fake_videos = fake_videos.view(self.config.batch_size_in_gpu, self.T, 3, 128, 128)
                #print("Saving image samples..")
                fake_videos = fake_videos.permute(0, 1, 3, 4, 2)
                tmpl = 'img_{:05d}.jpg'
                tmpl3 = '{:05d}.gif'
                tmpl2 = '{:05d}'
                sample_images = utils.denorm(fake_videos.detach()[:self.config.save_n_images])
                    
                for j in range(len(sample_images)):
                    video = sample_images[j].data.cpu().numpy()*255
                     number = (j+1) + (8 * gpu_batch)
                     gif_video = []                   
                     for i in range(video.shape[0]):
                        frame = Image.fromarray(video[i].astype('uint8'), 'RGB')
                        name = self.config.sample_images_path+'/ucf101_128_'+str(k)+'/class_'+str(label)+'/'+tmpl2.format(number)+'/'
                        utils.make_folder(name)
                        frame.save(name+tmpl.format(i+1))
                        gif_video.append(frame)
                        
                    imageio.mimsave(name+tmpl3.format(i), gif_video)
                            
                    with open(self.config.sample_images_path+'/ucf101_128_'+str(k)+'.txt', 'a') as f:
                        f.write(name+' '+str(video.shape[0])+' '+str(label)+'\n')
                    
                del fake_videos, sample_images, act, z
            print(label)

    def gen_z(self):
        z_C = torch.normal(mean=torch.full((self.config.batch_size_in_gpu, 80), 0), std=torch.full((self.config.batch_size_in_gpu, 80), 2)).cuda()
        z_S = torch.normal(mean=torch.full((self.config.batch_size_in_gpu, 20), 2), std=torch.full((self.config.batch_size_in_gpu, 20), 1)).cuda()
        z_M = torch.normal(mean=torch.full((self.config.batch_size_in_gpu, 20), 4), std=torch.full((self.config.batch_size_in_gpu, 20), 0.5)).cuda()
        
        z = torch.cat((z_M, z_C), 1)
        z = torch.cat((z, z_S), 1)
        return z.view(self.config.batch_size_in_gpu, self.config.z_dim).cuda()
    
        
    def build_models(self):
        self.G = Generator(self.config.z_dim, self.config.g_conv_dim, self.num_of_classes, 4).cuda()
        self.gru = CGRU(self.num_of_classes, self.config.z_dim, self.T, self.config.batch_size_in_gpu).cuda()

        self.G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.gru_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gru.parameters()), self.config.g_lr, [self.config.beta1, self.config.beta2])

        utils.load_pretrained_model(self, test=True)
        
        self.G = nn.DataParallel(self.G, device_ids=list(range(torch.cuda.device_count())))
        self.gru.train()

        print(self.G)
        print(self.gru)
    
    def Average(self, lst): 
        return sum(lst) / len(lst) 
 
from torch.nn.init import xavier_uniform_
from torch.nn.utils import spectral_norm
class CGRU(nn.Module):
    def __init__(self, num_classes, latent_size, T, batch):
        super(CGRU, self).__init__()
        self.latent_size = latent_size
        self.batch = batch
        self.fc = snlinear(latent_size, latent_size)
        self.gru = GRU(latent_size, 2048)
        self.gru.initWeight()

    def forward(self, z):
        h0 = self.fc(z)
        self.gru.initHidden(self.batch)
        h1 = self.gru(h0, 16).transpose(1, 0)
        return h1.contiguous().view(-1, h1.size(2))
def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))
