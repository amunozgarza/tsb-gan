import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_
import torch.nn.init as init


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snconv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, space=240):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, momentum=0.001, affine=False)
        self.gain = snlinear(in_features=space, out_features=num_features)
        self.bias = snlinear(in_features=space, out_features=num_features)

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        out = out * gain + bias
        return out


class InplaceShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fold):
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, space=256, fold=3):
        super(GenBlock, self).__init__()
        self.f=fold
        self.cond_bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.cond_bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, labels, shift):
        x0 = x
        if shift:
            x = self.shift(x, 16, fold_div=self.f, inplace=True)
        x = self.cond_bn1(x, labels)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.snconv2d1(x)
        x = self.cond_bn2(x, labels)
        x = self.relu(x)
        x = self.snconv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out
    
    def shift(self, x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        fold = c // fold_div
        if inplace:
            out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim, g_conv_dim, num_classes, fold):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.g_conv_dim = g_conv_dim
        self.snlinear0 = snlinear(in_features=z_dim+120, out_features=g_conv_dim*16*4*4)
        self.embed = sn_embedding(num_classes, 120)
        self.block1 = GenBlock(g_conv_dim*16, g_conv_dim*16, num_classes, fold=fold)
        self.block2 = GenBlock(g_conv_dim*16, g_conv_dim*8, num_classes, fold=fold)
        self.block3 = GenBlock(g_conv_dim*8, g_conv_dim*4, num_classes, fold=fold)
        self.self_attn = Self_Attn(g_conv_dim*4)
        self.block4 = GenBlock(g_conv_dim*4, g_conv_dim*2, num_classes, fold=fold)
        self.block5 = GenBlock(g_conv_dim*2, g_conv_dim, num_classes, fold=fold)
        self.bn = nn.BatchNorm2d(g_conv_dim, eps=1e-5, momentum=0.0001, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=g_conv_dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
        xavier_uniform_(self.embed.weight)

        # Weight init
        self.apply(init_weights)

    def forward(self, z, labels):
        # n x z_dim

        embed = self.embed(labels)
        labels_ = torch.cat((z, embed), 1)
        act0 = self.snlinear0(labels_)            # n x g_conv_dim*16*4*4
        act0 = act0.view(-1, self.g_conv_dim*16, 4, 4) # n x g_conv_dim*16 x 4 x 4
        act1 = self.block1(act0, labels_, True)    # n x g_conv_dim*16 x 8 x 8
        act2 = self.block2(act1, labels_, True)    # n x g_conv_dim*8 x 16 x 16
        act3 = self.block3(act2, labels_, False)    # n x g_conv_dim*4 x 32 x 32
        act3 = self.self_attn(act3)         # n x g_conv_dim*4 x 32 x 32
        act4 = self.block4(act3, labels_, False)    # n x g_conv_dim*2 x 64 x 64
        act5 = self.block5(act4, labels_, False)    # n x g_conv_dim  x 128 x 128
        act5 = self.bn(act5)                # n x g_conv_dim  x 128 x 128
        act5 = self.relu(act5)              # n x g_conv_dim  x 128 x 128
        act6 = self.snconv2d1(act5)         # n x 3 x 128 x 128
        act6 = self.tanh(act6)              # n x 3 x 128 x 128
        return act6

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if ('embed' not in name and 'gain' not in name):
                if ('bias' not in name and 'snlinear0' not in name):
                    if ('cond_bn1' not in name and 'cond_bn2' not in name):
                        param = param.data
                        own_state[name].copy_(param)


class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscOptBlock, self).__init__()
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x0 = x

        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.snconv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.snconv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.snconv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True
        self.snconv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.snconv2d1(x)
        x = self.relu(x)
        x = self.snconv2d2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.snconv2d0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, d_conv_dim, num_classes):
        super(Discriminator, self).__init__()
        self.d_conv_dim = d_conv_dim
        self.opt_block1 = DiscOptBlock(3, d_conv_dim)
        self.block1 = DiscBlock(d_conv_dim, d_conv_dim*2)
        self.self_attn = Self_Attn(d_conv_dim*2)
        self.block2 = DiscBlock(d_conv_dim*2, d_conv_dim*4)
        self.block3 = DiscBlock(d_conv_dim*4, d_conv_dim*8)
        self.block4 = DiscBlock(d_conv_dim*8, d_conv_dim*16)
        self.block5 = DiscBlock(d_conv_dim*16, d_conv_dim*16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=d_conv_dim*16, out_features=1)
        self.sn_embedding1 = sn_embedding(num_classes, d_conv_dim*16)

        # Weight init
        self.apply(init_weights)
        xavier_uniform_(self.sn_embedding1.weight)

    def forward(self, x, labels):
        # n x 3 x 128 x 128
        h0 = self.opt_block1(x) # n x d_conv_dim   x 64 x 64
        h1 = self.block1(h0)    # n x d_conv_dim*2 x 32 x 32
        h1 = self.self_attn(h1) # n x d_conv_dim*2 x 32 x 32
        h2 = self.block2(h1)    # n x d_conv_dim*4 x 16 x 16
        h3 = self.block3(h2)    # n x d_conv_dim*8 x  8 x  8
        h4 = self.block4(h3)    # n x d_conv_dim*16 x 4 x  4
        h5 = self.block5(h4, downsample=False)  # n x d_conv_dim*16 x 4 x 4
        h5 = self.relu(h5)              # n x d_conv_dim*16 x 4 x 4
        h6 = torch.sum(h5, dim=[2,3])   # n x d_conv_dim*16
        output1 = torch.squeeze(self.snlinear1(h6)) # n x 1
        # Projection
        h_labels = self.sn_embedding1(labels)   # n x d_conv_dim*16
        proj = torch.mul(h6, h_labels)          # n x d_conv_dim*16
        output2 = torch.sum(proj, dim=[1])      # n x 1
        # Out
        output = output1 + output2              # n x 1
        return output

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if 'embed' not in name:
                # backwards compatibility for serialized parameters
                param = param.data
                own_state[name].copy_(param)
        
class Discriminator_3D(nn.Module):
    def __init__(self, d_conv_dim, num_classes, attention, T=16):
        super(Discriminator_3D, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x T x 96 x 96
            snconv3d(3, d_conv_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x T/2 x 48 x 48
            snconv3d(d_conv_dim, d_conv_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(d_conv_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x T/4 x 24 x 24
            snconv3d(d_conv_dim * 2, d_conv_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(d_conv_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x T/8 x 12 x 12
            snconv3d(d_conv_dim * 4, d_conv_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm3d(d_conv_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x T/16  x 6 x 6
            )
        self.linear = snlinear(d_conv_dim * 8, 1)
        self.embed = sn_embedding(num_classes, d_conv_dim*8)
        # Weight init
        self.apply(init_weights)
        xavier_uniform_(self.embed.weight)
    def forward(self, input, class_id):
        output = self.main(input)
        output = torch.sum(output, dim=[3,4]).view(-1, output.size(1))
        output_linear = torch.squeeze(self.linear(output))
        y = class_id.long()
        embed = self.embed(y)
        prod = (output * embed).sum(1)
        return output_linear + prod#

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0, gpu=True):
        super(GRU, self).__init__()

        output_size      = input_size
        self._gpu        = gpu
        self.hidden_size = hidden_size

        # define layers
        self.gru    = nn.GRUCell(input_size, hidden_size)
        self.drop   = nn.Dropout(p=dropout)
        self.linear = snlinear(hidden_size, output_size)
        self.bn     = nn.BatchNorm1d(output_size, affine=False)

    def forward(self, inputs, n_frames):
        outputs = []
        for i in range(n_frames):
            self.hidden = self.gru(inputs, self.hidden)
            inputs = self.linear(self.hidden)
            outputs.append(inputs)
        outputs = [ self.bn(elm) for elm in outputs ]
        outputs = torch.stack(outputs)
        return outputs

    def initWeight(self, init_forget_bias=1):
        for name, params in self.named_parameters():
            if 'weight' in name:
                init.xavier_uniform(params)

            # initialize forget gate bias
            elif 'gru.bias_ih_l' in name:
                b_ir, b_iz, b_in = params.chunk(3, 0)
                init.constant(b_iz, init_forget_bias)
            elif 'gru.bias_hh_l' in name:
                b_hr, b_hz, b_hn = params.chunk(3, 0)
                init.constant(b_hz, init_forget_bias)
            else:
                init.constant(params, 0)

    def initHidden(self, batch_size):
        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        if self._gpu == True:
            self.hidden = self.hidden.cuda()
