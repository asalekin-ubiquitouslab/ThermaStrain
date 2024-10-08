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



       
class P2(nn.Module):
    def __init__(self,hyper_size,hidden_size,dropout,nchannels):
        super(P2, self).__init__()


        self.activation = nn.ReLU()

        self.feature_extractor = ResNet(BasicBlock, [2, 2, 2, 2],num_classes=hidden_size, input_channels=1,nchannels=nchannels)
        self.halluci =  Stander_Encoder(hidden_size, hidden_size,n_layers=1,n_head=2)


        self.fusion = nn.Sequential()
        
        #self.fusion.add_module('norm', nn.LayerNorm(hyper_size))
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=hyper_size, out_features=hidden_size))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.fusion.add_module('fusion_layer_2', nn.Linear(in_features=hidden_size, out_features=2))
        self.fusion.add_module('fusion_layer_2_activation', self.activation)
        self.fusion.add_module('fusion_layer_2_dropout', nn.Dropout(dropout))

        self.eda1 = nn.Sequential()
        self.eda1.add_module('fusion_layer_1', nn.Linear(in_features=6, out_features=hidden_size))
        self.eda1.add_module('fusion_layer_1_activation', self.activation)
        self.eda1.add_module('fusion_layer_1_dropout', nn.Dropout(dropout))
        self.eda1.add_module('fusion_layer_2', nn.Linear(in_features=hidden_size, out_features=hyper_size))
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
            nn.Linear(hidden_size, hyper_size),
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
            loss_recon+=0.5*self.diver(F.softmax(self.o, dim=1),F.softmax(self.eda, dim=1))

            loss=loss+loss_recon+loss2

            y_hat=F.softmax(fake,dim=1)
            y_hat = torch.max(y_hat,1)[1]

            return loss,y_hat,logit

