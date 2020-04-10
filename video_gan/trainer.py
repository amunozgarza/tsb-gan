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
import GPUtil as GPU
import cv2

from torch.backends import cudnn

import utils
from models import Generator, Discriminator, GRU, Discriminator_3D
from dataset import TSNDataSet


class Trainer(object):

    def __init__(self, config):

        # Config
        self.config = config

        self.start = 0 # Unless using pre-trained model

        # Create directories if not exist
        utils.make_folder(self.config.save_path)
        utils.make_folder(self.config.model_weights_path)
        utils.make_folder(self.config.sample_images_path)
        utils.make_folder(self.config.log_path)

        self.T = 16

        cropping_size = (self.config.imsize, self.config.imsize)
        # Check for CUDA
        utils.check_for_CUDA(self)
        
        if self.config.dataset == "penn_action":
            tmpl = '{:06d}.jpg'
        else:
            tmpl = 'img_{:05d}.jpg'
        
        if config.dataset == "weizmann":
            data_loader = torch.utils.data.DataLoader(TSNDataSet("", self.config, self.config.data_path, num_segments=self.T,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.CenterCrop(cropping_size),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                       
                           ]),), batch_size=config.batch_size_in_gpu, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True)
        else:
            data_loader = torch.utils.data.DataLoader(TSNDataSet("", self.config, self.config.data_path, num_segments=self.T, image_tmpl= tmpl,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.Resize((170, 170)),
                               torchvision.transforms.CenterCrop(cropping_size),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                       
                           ]),), batch_size=config.batch_size_in_gpu, shuffle=True,
                num_workers=4, pin_memory=True, drop_last=True)

        # Make dataloader
        self.dataloader = data_loader

        if self.config.dataset == "weizmann":
            self.num_of_classes = 10
        if self.config.dataset == "ucf11":
            self.num_of_classes = 11
        if self.config.dataset == "ucf101":
            self.num_of_classes = 101
        if self.config.dataset == "kinetics":
            self.num_of_classes = 700
        if self.config.dataset == "penn_action":
            self.num_of_classes = 14
        # Data iterator
        self.data_iter = iter(self.dataloader)

        # Build G and D
        self.build_models()

    def train(self):

        # Seed
        np.random.seed(self.config.manual_seed)
        random.seed(self.config.manual_seed)
        torch.manual_seed(self.config.manual_seed)

        # For fast training
        cudnn.benchmark = True

        # For BatchNorm
        self.G.train()
        self.D.train()
        self.D_3D.train()
        self.gru.train()

        # Fixed noise for sampling from G
        fixed_noise = self.gen_z(self.config.save_n_images)
        if self.num_of_classes < self.config.batch_size_in_gpu:
            fixed_labels = torch.from_numpy(np.tile(np.arange(self.num_of_classes), self.config.batch_size_in_gpu//self.num_of_classes + 1)[:self.config.batch_size_in_gpu]).to(self.device)
        else:
            fixed_labels = torch.from_numpy(np.arange(self.config.save_n_images)).to(self.device)
        fixed_labels = fixed_labels.data.cpu().numpy()
        fixed_labels = fixed_labels.repeat(self.T, axis=0)
        fixed_labels = torch.from_numpy(fixed_labels).to(self.device)

        # For gan loss
        label = torch.full((self.config.batch_size_in_gpu,), 1, device=self.device)
        ones = torch.full((self.config.batch_size_in_gpu,), 1, device=self.device)
        ones_v = torch.full((self.config.batch_size_in_gpu*8,), 1, device=self.device)

        # Losses file
        log_file_name = os.path.join(self.config.log_path, 'log.txt')
        log_file = open(log_file_name, "wt")

        # Init
        start_time = time.time()
        start_time2 = time.time()
        G_losses_i = []
        G_losses_v = []
        D_losses_real_i = []
        D_losses_fake_i = []
        D_losses_real_v = []
        D_losses_fake_v = []
        D_losses_i = []
        D_losses_v = []

        # Instance noise - make random noise mean (0) and std for injecting
        inst_noise_mean = torch.full((self.config.batch_size_in_gpu, 3, 8, self.config.imsize, self.config.imsize), 0, device=self.device)
        inst_noise_std = torch.full((self.config.batch_size_in_gpu, 3, 8, self.config.imsize, self.config.imsize), self.config.inst_noise_sigma, device=self.device)
        
        inst_noise_mean_v = torch.full((self.config.batch_size_in_gpu, 3, self.T, self.config.imsize, self.config.imsize), 0, device=self.device)
        inst_noise_std_v = torch.full((self.config.batch_size_in_gpu, 3, self.T, self.config.imsize, self.config.imsize), self.config.inst_noise_sigma, device=self.device)

        self.gpu_batches = self.config.batch_size//self.config.batch_size_in_gpu

        # Start training
        for self.step in range(self.start, self.config.total_step):

            # Instance noise std is linearly annealed from self.inst_noise_sigma to 0 thru self.inst_noise_sigma_iters
            inst_noise_sigma_curr = 0 if self.step > self.config.inst_noise_sigma_iters else (1 - self.step/self.config.inst_noise_sigma_iters)*self.config.inst_noise_sigma
            inst_noise_std.fill_(inst_noise_sigma_curr)
            inst_noise_std_v.fill_(inst_noise_sigma_curr)
            

            # ================== TRAIN D ================== #
            loss_real_i = []
            loss_fake_i = []
            loss_real_v = []
            loss_fake_v = []
            for _ in range(self.config.d_steps_per_iter):

                # Zero grad
                self.reset_grad()
                # Accumulate losses for full batch_size
                # while running GPU computations on only batch_size_in_gpu
                for gpu_batch in range(self.gpu_batches):
                    # TRAIN with REAL

                    # Get real videos & real labels
                    real_videos, real_labels = self.get_real_samples()
                    real_images, real_labels_v = self.sample_frames(real_videos, real_labels)
                    
                    # Get D output for real images & real labels
                    inst_noise = torch.normal(mean=inst_noise_mean, std=inst_noise_std).to(self.device)

                    real_images = real_images.transpose(2,1).contiguous().view(self.config.batch_size_in_gpu*8, 3, self.config.imsize, self.config.imsize)
                    d_out_real_i = self.D(real_images.float() + inst_noise.contiguous().view(self.config.batch_size_in_gpu*8, 3, self.config.imsize, self.config.imsize), 
                        real_labels_v)

                    # Compute D loss with real images & real labels
                    if self.config.adv_loss == 'hinge':
                        d_loss_real_i = torch.nn.ReLU()(ones_v - d_out_real_i).mean()

                    # Backward
                    d_loss_real_i /= self.gpu_batches
                    d_loss_real_i.backward()
                    loss_real_i.append(d_loss_real_i.detach().cpu().numpy())

                    # Delete loss, output
                    if self.step % self.config.log_step != 0 or gpu_batch < self.gpu_batches - 1:
                        del d_out_real_i, d_loss_real_i

                    # TRAIN with FAKE

                    # Create random noise

                    z = self.gen_z(self.config.batch_size_in_gpu)

                    fake_labels, fake_labels_i, fake_labels_v = self.samples_fake_labels(real_labels)
                    
                    # Generate fake videos
                    act = self.gru(z)
                    fake_videos = self.G(act, fake_labels_v)
                    fake_videos = fake_videos.view(self.config.batch_size_in_gpu, self.T, 3, self.config.imsize, self.config.imsize)
                    fake_videos = fake_videos.transpose(2, 1)
                    
                    fake_images, _ = self.sample_frames(fake_videos, fake_labels) 
                    fake_images = fake_images.transpose(2,1).contiguous().view(self.config.batch_size_in_gpu*8, 3, self.config.imsize, self.config.imsize)

                    inst_noise = torch.normal(mean=inst_noise_mean, std=inst_noise_std).to(self.device)
                    
                    d_out_fake_i = self.D(fake_images.detach() + inst_noise.contiguous().view(self.config.batch_size_in_gpu*8, 3, self.config.imsize, self.config.imsize), 
                        fake_labels_i.squeeze())

                    # Compute D loss with fake images & real labels
                    if self.config.adv_loss == 'hinge':
                        d_loss_fake_i = torch.nn.ReLU()(ones_v + d_out_fake_i).mean()

                    # If WGAN_GP, compute GP and add to D loss
                    if self.config.adv_loss == 'wgan_gp':
                        d_loss_gp = self.config.lambda_gp * self.compute_gradient_penalty(real_images, real_labels, fake_images.detach())
                        d_loss_fake += d_loss_gp

                    # Backward
                    d_loss_fake_i /= self.gpu_batches
                    d_loss_fake_i.backward()
                    loss_fake_i.append(d_loss_fake_i.detach().cpu().numpy())

                    # Delete loss, output
                    #del fake_images
                    if self.step % self.config.log_step != 0 or gpu_batch < self.gpu_batches - 1:
                        del d_out_fake_i, d_loss_fake_i
                    
                    #TRAIN REAL VIDEOS
                    inst_noise = torch.normal(mean=inst_noise_mean_v, std=inst_noise_std_v).to(self.device)
                    d_out_real_v = self.D_3D(real_videos.float() + inst_noise, real_labels)
                    
                    if self.config.adv_loss == 'hinge':
                        d_loss_real_v = torch.nn.ReLU()(ones - d_out_real_v).mean()
                    
                    d_loss_real_v /= self.gpu_batches
                    d_loss_real_v.backward()
                    loss_real_v.append(d_loss_real_v.detach().cpu().numpy())
                    
                    if self.step % self.config.log_step != 0 or gpu_batch < self.gpu_batches - 1:
                        del d_out_real_v, d_loss_real_v
                        
                    #TRAIN FAKE VIDEOS
                    inst_noise = torch.normal(mean=inst_noise_mean_v, std=inst_noise_std_v).to(self.device)
                    d_out_fake_v = self.D_3D(fake_videos.detach() + inst_noise, fake_labels.squeeze())
                    
                    if self.config.adv_loss == 'hinge':
                        d_loss_fake_v = torch.nn.ReLU()(ones + d_out_fake_v).mean()
                    
                    d_loss_fake_v /= self.gpu_batches
                    d_loss_fake_v.backward()
                    loss_fake_v.append(d_loss_fake_v.detach().cpu().numpy())
                    # Delete loss, output
                    del fake_videos
                    del fake_images
                    if self.step % self.config.log_step != 0 or gpu_batch < self.gpu_batches - 1:
                        del d_out_fake_v, d_loss_fake_v

                # Optimize
                self.D_optimizer.step()
                self.D_3D_optimizer.step()

            # ================== TRAIN G ================== #
            loss_g_i = []
            loss_g_v = []

            for _ in range(self.config.g_steps_per_iter):

                # Zero grad
                self.reset_grad()

                # Accumulate losses for full batch_size
                # while running GPU computations on only batch_size_in_gpu
                for gpu_batch in range(self.gpu_batches):

                    # Get real videos & real labels (only need real labels)
                    real_videos, real_labels = self.get_real_samples()
                    real_images = real_videos[:, :, np.random.randint(0, self.T), :, :]

                    # Create random noise
                    z = self.gen_z(self.config.batch_size_in_gpu)

                    fake_labels, fake_labels_i , fake_labels_v = self.samples_fake_labels(real_labels)

                    act = self.gru(z)
                    # Generate fake images for same real labels
                    fake_videos = self.G(act, fake_labels_v)
                    fake_videos = fake_videos.view(self.config.batch_size_in_gpu, self.T, 3, self.config.imsize, self.config.imsize)
                    fake_videos = fake_videos.transpose(2, 1)
                    
                    fake_images, _ = self.sample_frames(fake_videos, fake_labels)
                    
                    fake_images = fake_images.transpose(2,1).contiguous().view(self.config.batch_size_in_gpu*8, 3, self.config.imsize, self.config.imsize)

                    inst_noise = torch.normal(mean=inst_noise_mean, std=inst_noise_std).to(self.device)
                    g_out_fake_i = self.D(fake_images + inst_noise.contiguous().view(self.config.batch_size_in_gpu*8, 3, self.config.imsize, self.config.imsize), fake_labels_i.squeeze())

                    # Compute G loss with fake images & real labels
                    g_loss_i = -g_out_fake_i.mean()

                    # Backward
                    g_loss_i /= self.gpu_batches
                    g_loss_i.backward(retain_graph=True)
                    loss_g_i.append(g_loss_i.detach().cpu().numpy())

                    # Delete loss, output
                    if self.step % self.config.log_step != 0 or gpu_batch < self.gpu_batches - 1:
                        del g_out_fake_i, g_loss_i
                        
                    #TRAIN VIDEOS
                    inst_noise = torch.normal(mean=inst_noise_mean_v, std=inst_noise_std_v).to(self.device)
                    g_out_fake_v = self.D_3D(fake_videos + inst_noise, fake_labels.squeeze())

                    # Compute G loss with fake images & real labels
                    g_loss_v = -g_out_fake_v.mean()

                    # Backward
                    g_loss_v /= self.gpu_batches
                    g_loss_v.backward()
                    loss_g_v.append(g_loss_v.detach().cpu().numpy())

                    # Delete loss, output
                    del fake_images
                    del fake_videos
                    if self.step % self.config.log_step != 0 or gpu_batch < self.gpu_batches - 1:
                        del g_out_fake_v, g_loss_v

                # Optimize
                self.G_optimizer.step()
                self.gru_optimizer.step()
                

            # Print out log info
            if self.step % self.config.log_step == 0:
                G_losses_i.append(self.Average(loss_g_i))
                D_losses_real_i.append(self.Average(loss_real_i))
                D_losses_fake_i.append(self.Average(loss_fake_i))
                D_loss_i = D_losses_real_i[-1] + D_losses_fake_i[-1]
                G_losses_v.append(self.Average(loss_g_v))
                D_losses_real_v.append(self.Average(loss_real_v))
                D_losses_fake_v.append(self.Average(loss_fake_v))
                D_loss_v = D_losses_real_v[-1] + D_losses_fake_v[-1]
                if self.config.adv_loss == 'wgan_gp':
                    D_loss += d_loss_gp.mean().item()
                D_losses_i.append(D_loss_i)
                D_losses_v.append(D_loss_v)
                
                curr_time = time.time()
                curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
                elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
                #print(D_losses_v)
                log = ("[{}]:Elapsed [{}], Iter [{} / {}], G_loss_i: {:.4f}, D_loss_i: {:.4f}, D_loss_real_i: {:.4f}, D_loss_fake_i: {:.4f}, G_loss_v: {:.4f}, D_loss_v: {:.4f}, D_loss_real_v: {:.4f}, D_loss_fake_v: {:.4f} \n" .
                       format(curr_time_str, elapsed, self.step, self.config.total_step,
                              G_losses_i[-1], D_losses_i[-1], D_losses_real_i[-1], D_losses_fake_i[-1],
                              G_losses_v[-1], D_losses_v[-1], D_losses_real_v[-1], D_losses_fake_v[-1]))
                print('\n' + log)
                log_file.write(log)
                log_file.flush()
                with open(os.path.join(self.config.log_path, 'log_graph.txt'), 'a') as f:
                    f.write(str(G_losses_i[-1])+' '+str(D_losses_i[-1])+' '+str(D_losses_real_i[-1])+' '+str(D_losses_fake_i[-1])+' '+
                              str(G_losses_v[-1])+' '+str(D_losses_v[-1])+' '+str(D_losses_real_v[-1])+' '+str(D_losses_fake_v[-1])+'\n')

                # Delete loss, output
                del d_out_real_i, d_loss_real_i, d_out_fake_i, d_loss_fake_i, g_out_fake_i, g_loss_i
                del d_out_real_v, d_loss_real_v, d_out_fake_v, d_loss_fake_v, g_out_fake_v, g_loss_v

            # Sample images
            if self.step % self.config.sample_step == 0:
                print("Saving image samples..")
                self.G.eval()
                act = self.gru(fixed_noise)
                fake_videos = self.G(act, fixed_labels)
                fake_videos2 = fake_videos.detach().cpu()
                fake_videos = fake_videos.view(self.config.save_n_images, self.T, 3, self.config.imsize, self.config.imsize)
                
                fake_videos = fake_videos.permute(0, 1, 3, 4, 2)

                GPU.showUtilization(all=True)
                
                self.G.train()
                sample_images = utils.denorm(fake_videos.detach()[:self.config.save_n_images])
                sample_images2 = utils.denorm(fake_videos2[:self.config.save_n_images*self.T])
                utils.make_folder(self.config.sample_images_path+'/step_'+str(self.step))
                for j in range(len(sample_images)):
                    video = sample_images[j].data.cpu().numpy()*255
                    pil_vid = []
                    
                    for i in range(video.shape[0]):
                        pil_vid.append(Image.fromarray(video[i].astype('uint8'), 'RGB'))
                    
                    name = self.config.sample_images_path+'/step_'+str(self.step)+'/step_'+str(self.step)+'_video_'+str(j)+'.gif'
                    pil_vid[0].save(name, format='GIF', append_images=pil_vid[1:], save_all=True, duration=100, loop=0)
                # Save batch images
                vutils.save_image(sample_images2, os.path.join(self.config.sample_images_path+'/step_'+str(self.step)+'/', 'fake_{:05d}.png'.format(self.step)), nrow=self.config.nrow)
               
                del fake_videos, fake_videos2

            # Save model
            if ((time.time() - start_time2)/60/60) > 22:
                t = int(self.config.model_save_step/100)
            else:
                t = 1
            if (self.step % (self.config.model_save_step/t) == 0) and (self.step != 0):
                utils.save_ckpt(self)

    def gen_z(self, number):
        z_C = torch.normal(mean=torch.full((number, 80), 0), std=torch.full((number, 80), 2)).to(self.device)
        z_S = torch.normal(mean=torch.full((number, 20), 0), std=torch.full((number, 20), 1)).to(self.device)
        z_M = torch.normal(mean=torch.full((number, 20), 0), std=torch.full((number, 20), 0.5)).to(self.device)
        
        z = torch.cat((z_M, z_C), 1)
        z = torch.cat((z, z_S), 1)

        return z.view(number, self.config.z_dim).to(self.device)
        
    def build_models(self):
        self.G = Generator(self.config.z_dim, self.config.g_conv_dim, self.num_of_classes, self.config.fold).to(self.device)
        self.D = Discriminator(self.config.d_conv_dim, self.num_of_classes).to(self.device)
        self.D_3D = Discriminator_3D(self.config.d_conv_dim, self.num_of_classes, self.config.attention, T=self.T).to(self.device)
        self.gru = CGRU(self.num_of_classes, self.config.z_dim, self.T, self.config.batch_size_in_gpu).to(self.device)

        # Loss and optimizer
        self.G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.config.g_lr, [self.config.beta1, self.config.beta2])
        self.D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.config.d_lr, [self.config.beta1, self.config.beta2])
        self.D_3D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D_3D.parameters()), self.config.d_lr, [self.config.beta1, self.config.beta2])
        self.gru_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gru.parameters()), self.config.g_lr, [self.config.beta1, self.config.beta2])

        # Start with pretrained model (if it exists)
        utils.load_pretrained_model(self)
        
        if self.config.pretrained and self.start == 0:
            checkpoint = torch.load(self.config.pretrained_path)
            self.G.load_my_state_dict(checkpoint['G_state_dict'])
            self.D.load_my_state_dict(checkpoint['D_state_dict'])

        self.G = nn.DataParallel(self.G, device_ids=list(range(torch.cuda.device_count())))
        self.D = nn.DataParallel(self.D, device_ids=list(range(torch.cuda.device_count())))
        self.D_3D = nn.DataParallel(self.D_3D, device_ids=list(range(torch.cuda.device_count())))

        # print networks
        print(self.G)
        print(self.D)
        print(self.D_3D)
        print(self.gru)
    

    def reset_grad(self):
        self.G_optimizer.zero_grad()
        self.D_optimizer.zero_grad()
        self.D_3D_optimizer.zero_grad()
        self.gru_optimizer.zero_grad()
    
    def get_real_samples(self):
        try:
            real_images, real_labels = next(self.data_iter)
        except:
            self.data_iter = iter(self.dataloader)
            real_images, real_labels = next(self.data_iter)

        real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
        return real_images, real_labels
    
    def Average(self, lst): 
        return sum(lst) / len(lst)

    def sample_frames(self, real_videos, real_labels):
        real_images = torch.cuda.FloatTensor(self.config.batch_size_in_gpu, 3, 8, self.config.imsize, self.config.imsize)
        for i in range(self.config.batch_size_in_gpu):
            frames = np.random.randint(0, self.T, size=8)
            for j, idx in enumerate(frames):
                real_images[i, :, j, :, :] = real_videos[i, :, idx, :, :]
                real_labels_v = real_labels.data.cpu().numpy()
                #print(real_labels_v.shape)
                real_labels_v = np.reshape(real_labels_v, (self.config.batch_size_in_gpu, 1))
                real_labels_v = real_labels_v.repeat(8, axis=1)
                real_labels_v = torch.from_numpy(real_labels_v).contiguous().view(self.config.batch_size_in_gpu*8).to(self.device).squeeze()
        return real_images, real_labels_v

    def samples_fake_labels(self, real_labels):
        real_labels_v = real_labels.data.cpu().numpy()
        fake_labels = np.random.randint(0, self.num_of_classes, real_labels_v.shape)
        fake_labels = np.reshape(fake_labels, (self.config.batch_size_in_gpu, 1))
        fake_labels_v = fake_labels.repeat(self.T, axis=1)
        fake_labels_v = torch.from_numpy(fake_labels_v).to(self.device).view(self.config.batch_size_in_gpu*self.T)
        fake_labels_i = fake_labels.repeat(8, axis=1)
        fake_labels_i = torch.from_numpy(fake_labels_i).to(self.device).contiguous().view(self.config.batch_size_in_gpu*8)
        fake_labels = torch.from_numpy(fake_labels).to(self.device)
        return fake_labels, fake_labels_i, fake_labels_v

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
