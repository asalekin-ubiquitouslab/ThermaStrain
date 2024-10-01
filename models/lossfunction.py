import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
from torch.autograd import Variable


from torch.distributions import Normal
from models.resnet import ResNet,BasicBlock
from models.stander import Stander_Encoder
import torch
from torch import nn
class CMD(nn.Module):


    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=3):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
 


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class flow2KL(nn.Module):
    def __init__(self,hyper_size,hidden_size,dropout,nchannels):
        super(flow2KL, self).__init__()


        self.activation = nn.ReLU()

        self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],num_classes=hidden_size, input_channels=1)
        self.halluci =  Stander_Encoder(hidden_size, hidden_size,n_layers=1,n_head=2)


        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_norm', nn.LayerNorm(32))
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=32, out_features=hidden_size))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.fusion.add_module('fusion_layer_2', nn.Linear(in_features=hidden_size, out_features=2))
        self.fusion.add_module('fusion_layer_2_activation', self.activation)
        self.fusion.add_module('fusion_layer_2_dropout', nn.Dropout(dropout))

        self.eda1 = nn.Sequential()
        self.eda1.add_module('fusion_layer_1', nn.Linear(in_features=6, out_features=hidden_size))
        self.eda1.add_module('fusion_layer_1_activation', self.activation)
        self.eda1.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.eda1.add_module('fusion_layer_2', nn.Linear(in_features=hidden_size, out_features=32))
        self.eda1.add_module('fusion_layer_2_activation', self.activation)
        self.eda1.add_module('fusion_layer_2_dropout', nn.Dropout(dropout))
        
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        
        nsize=self._get_nsize((10,1, 240, 320))
        
        
        self.embedding = nn.Sequential()
        self.embedding.add_module('fusion_layer_1', nn.Linear(in_features=nsize, out_features=hidden_size))
        self.embedding.add_module('fusion_layer_1_activation', self.activation)
        self.embedding.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        
        self.encoder= nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.Dropout(p=dropout),
            nn.ReLU(),

        )
        

        self.loss_class = nn.CrossEntropyLoss()
        self.loss_recon=nn.MSELoss()
        self.diver=nn.KLDivLoss(reduction='batchmean')

        
    def _get_nsize(self, shape):
        x = Variable(torch.rand(*shape))
        x = self.feature_extractor(x)
        x = self.avgpool(x)
   
        n_size = x.view(shape[0], -1).size(1) #batch, nsize
        return n_size
    def forward(self,  frame,eda):
        frame=frame.squeeze().unsqueeze(2)
        batch_size,time_steps,channel,H,W = frame.shape
        thermal=frame.reshape(batch_size*time_steps,channel,H,W)

        h=self.feature_extractor(thermal)
        h=self.avgpool(h).reshape(batch_size,time_steps,-1)
        h=self.embedding(h)
        h=self.halluci(h)
        h=torch.mean(h,1)
        self.h=self.encoder(h)

        self.h_eda=self.eda1(eda)

  
        self.o = self.fusion(self.h)
        self.eda = self.fusion(self.h_eda)

        return self.o,self.eda
        
    def infer(self,  frame,eda):
        

        frame=frame.squeeze().unsqueeze(2)
        batch_size,time_steps,channel,H,W = frame.shape
        thermal=frame.reshape(batch_size*time_steps,channel,H,W)

        h=self.feature_extractor(thermal)
        h=self.avgpool(h).reshape(batch_size,time_steps,-1)
        h=self.embedding(h)
        
        h=self.halluci(h)
        h=torch.mean(h,1)
        self.h=self.encoder(h)

  
        self.o = self.fusion(self.h)
        

        return self.o
    def get_loss(self,  outputs,label):
        if len(outputs)!=2:
            fake=outputs
            logit=F.softmax(fake,dim=1)[:,1].tolist()

            label=label.view(-1)
            loss = self.loss_class(fake, label)


            y_hat=F.softmax(fake,dim=1)
            y_hat = torch.max(y_hat,1)[1]

            return loss,y_hat,logit
        else:

            fake,real=outputs
            logit=F.softmax(fake,dim=1)[:,1].tolist()

            label=label.view(-1)
            loss = self.loss_class(fake, label)
            loss2 = self.loss_class(real, label)


            loss_recon = 0.5*self.loss_recon((self.h), (self.h_eda))
            #loss_recon += 0.5*self.loss_recon((self.psy_recon), (self.raw_psy))
            #loss_recon+=0.5*self.diver(F.softmax(self.o, dim=1),F.softmax(self.eda, dim=1))

            loss=loss+loss_recon+loss2

            y_hat=F.softmax(fake,dim=1)
            y_hat = torch.max(y_hat,1)[1]

            return loss,y_hat,logit
            
           
class flow2cmd(nn.Module):
    def __init__(self,hyper_size,hidden_size,dropout,nchannels):
        super(flow2cmd, self).__init__()


        self.activation = nn.ReLU()

        self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],num_classes=hidden_size, input_channels=1)
        self.halluci =  Stander_Encoder(hidden_size, hidden_size,n_layers=1,n_head=2)


        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_norm', nn.LayerNorm(32))
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=32, out_features=hidden_size))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.fusion.add_module('fusion_layer_2', nn.Linear(in_features=hidden_size, out_features=2))
        self.fusion.add_module('fusion_layer_2_activation', self.activation)
        self.fusion.add_module('fusion_layer_2_dropout', nn.Dropout(dropout))

        self.eda1 = nn.Sequential()
        self.eda1.add_module('fusion_layer_1', nn.Linear(in_features=6, out_features=hidden_size))
        self.eda1.add_module('fusion_layer_1_activation', self.activation)
        self.eda1.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.eda1.add_module('fusion_layer_2', nn.Linear(in_features=hidden_size, out_features=32))
        self.eda1.add_module('fusion_layer_2_activation', self.activation)
        self.eda1.add_module('fusion_layer_2_dropout', nn.Dropout(dropout))
        
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        
        nsize=self._get_nsize((10,1, 240, 320))
        
        
        self.embedding = nn.Sequential()
        self.embedding.add_module('fusion_layer_1', nn.Linear(in_features=nsize, out_features=hidden_size))
        self.embedding.add_module('fusion_layer_1_activation', self.activation)
        self.embedding.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        
        self.encoder= nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.Dropout(p=dropout),
            nn.ReLU(),

        )
        

        self.loss_class = nn.CrossEntropyLoss()
        self.loss_recon=CMD()
        self.diver=nn.KLDivLoss(reduction='batchmean')

        
    def _get_nsize(self, shape):
        x = Variable(torch.rand(*shape))
        x = self.feature_extractor(x)
        x = self.avgpool(x)
   
        n_size = x.view(shape[0], -1).size(1) #batch, nsize
        return n_size
    def forward(self,  frame,eda):
        frame=frame.squeeze().unsqueeze(2)
        batch_size,time_steps,channel,H,W = frame.shape
        thermal=frame.reshape(batch_size*time_steps,channel,H,W)

        h=self.feature_extractor(thermal)
        h=self.avgpool(h).reshape(batch_size,time_steps,-1)
        h=self.embedding(h)
        h=self.halluci(h)
        h=torch.mean(h,1)
        self.h=self.encoder(h)

        self.h_eda=self.eda1(eda)

  
        self.o = self.fusion(self.h)
        self.eda = self.fusion(self.h_eda)

        return self.o,self.eda
        
    def infer(self,  frame,eda):
        

        frame=frame.squeeze().unsqueeze(2)
        batch_size,time_steps,channel,H,W = frame.shape
        thermal=frame.reshape(batch_size*time_steps,channel,H,W)

        h=self.feature_extractor(thermal)
        h=self.avgpool(h).reshape(batch_size,time_steps,-1)
        h=self.embedding(h)
        
        h=self.halluci(h)
        h=torch.mean(h,1)
        self.h=self.encoder(h)

  
        self.o = self.fusion(self.h)
        

        return self.o
    def get_loss(self,  outputs,label):
        if len(outputs)!=2:
            fake=outputs
            logit=F.softmax(fake,dim=1)[:,1].tolist()

            label=label.view(-1)
            loss = self.loss_class(fake, label)


            y_hat=F.softmax(fake,dim=1)
            y_hat = torch.max(y_hat,1)[1]

            return loss,y_hat,logit
        else:

            fake,real=outputs
            logit=F.softmax(fake,dim=1)[:,1].tolist()

            label=label.view(-1)
            loss = self.loss_class(fake, label)
            loss2 = self.loss_class(real, label)


            loss_recon = 0.5*self.loss_recon(F.softmax(self.h, dim=1), F.softmax(self.h_eda, dim=1))
            #loss_recon += 0.5*self.loss_recon((self.psy_recon), (self.raw_psy))
            loss_recon+=0.5*self.diver(F.softmax(self.o, dim=1),F.softmax(self.eda, dim=1))

            loss=loss+loss_recon+loss2

            y_hat=F.softmax(fake,dim=1)
            y_hat = torch.max(y_hat,1)[1]

            return loss,y_hat,logit
            
           
class flow2mmd(nn.Module):
    def __init__(self,hyper_size,hidden_size,dropout,nchannels):
        super(flow2mmd, self).__init__()


        self.activation = nn.ReLU()

        self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],num_classes=hidden_size, input_channels=1)
        self.halluci =  Stander_Encoder(hidden_size, hidden_size,n_layers=1,n_head=2)


        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_norm', nn.LayerNorm(32))
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=32, out_features=hidden_size))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.fusion.add_module('fusion_layer_2', nn.Linear(in_features=hidden_size, out_features=2))
        self.fusion.add_module('fusion_layer_2_activation', self.activation)
        self.fusion.add_module('fusion_layer_2_dropout', nn.Dropout(dropout))

        self.eda1 = nn.Sequential()
        self.eda1.add_module('fusion_layer_1', nn.Linear(in_features=6, out_features=hidden_size))
        self.eda1.add_module('fusion_layer_1_activation', self.activation)
        self.eda1.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.eda1.add_module('fusion_layer_2', nn.Linear(in_features=hidden_size, out_features=32))
        self.eda1.add_module('fusion_layer_2_activation', self.activation)
        self.eda1.add_module('fusion_layer_2_dropout', nn.Dropout(dropout))
        
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        
        nsize=self._get_nsize((10,1, 240, 320))
        
        
        self.embedding = nn.Sequential()
        self.embedding.add_module('fusion_layer_1', nn.Linear(in_features=nsize, out_features=hidden_size))
        self.embedding.add_module('fusion_layer_1_activation', self.activation)
        self.embedding.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        
        self.encoder= nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.Dropout(p=dropout),
            nn.ReLU(),

        )
        

        self.loss_class = nn.CrossEntropyLoss()
        self.loss_recon=MMD_loss()
        self.diver=nn.KLDivLoss(reduction='batchmean')

        
    def _get_nsize(self, shape):
        x = Variable(torch.rand(*shape))
        x = self.feature_extractor(x)
        x = self.avgpool(x)
   
        n_size = x.view(shape[0], -1).size(1) #batch, nsize
        return n_size
    def forward(self,  frame,eda):
        frame=frame.squeeze().unsqueeze(2)
        batch_size,time_steps,channel,H,W = frame.shape
        thermal=frame.reshape(batch_size*time_steps,channel,H,W)

        h=self.feature_extractor(thermal)
        h=self.avgpool(h).reshape(batch_size,time_steps,-1)
        h=self.embedding(h)
        h=self.halluci(h)
        h=torch.mean(h,1)
        self.h=self.encoder(h)

        self.h_eda=self.eda1(eda)

  
        self.o = self.fusion(self.h)
        self.eda = self.fusion(self.h_eda)

        return self.o,self.eda
        
    def infer(self,  frame,eda):
        

        frame=frame.squeeze().unsqueeze(2)
        batch_size,time_steps,channel,H,W = frame.shape
        thermal=frame.reshape(batch_size*time_steps,channel,H,W)

        h=self.feature_extractor(thermal)
        h=self.avgpool(h).reshape(batch_size,time_steps,-1)
        h=self.embedding(h)
        
        h=self.halluci(h)
        h=torch.mean(h,1)
        self.h=self.encoder(h)

  
        self.o = self.fusion(self.h)
        

        return self.o
    def get_loss(self,  outputs,label):
        if len(outputs)!=2:
            fake=outputs
            logit=F.softmax(fake,dim=1)[:,1].tolist()

            label=label.view(-1)
            loss = self.loss_class(fake, label)


            y_hat=F.softmax(fake,dim=1)
            y_hat = torch.max(y_hat,1)[1]

            return loss,y_hat,logit
        else:

            fake,real=outputs
            logit=F.softmax(fake,dim=1)[:,1].tolist()

            label=label.view(-1)
            loss = self.loss_class(fake, label)
            loss2 = self.loss_class(real, label)


            loss_recon = 0.5*self.loss_recon(F.softmax(self.h, dim=1), F.softmax(self.h_eda, dim=1))
            #loss_recon += 0.5*self.loss_recon((self.psy_recon), (self.raw_psy))
            loss_recon+=0.5*self.diver(F.softmax(self.o, dim=1),F.softmax(self.eda, dim=1))

            loss=loss+loss_recon+loss2

            y_hat=F.softmax(fake,dim=1)
            y_hat = torch.max(y_hat,1)[1]

            return loss,y_hat,logit