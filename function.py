import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import json
from tqdm import tqdm
import glob
from torchvision import transforms  
import h5py

import time
def EDAanalysis(EDAdata):
    featurelist=[]
    featurelist.append(np.mean(EDAdata))
    featurelist.append(np.amin(EDAdata))
    featurelist.append(np.amax(EDAdata))
    featurelist.append(np.median(EDAdata))
    featurelist.append(np.var(EDAdata))
    featurelist.append(np.std(EDAdata))
    featurelist=np.stack(featurelist)
    
    return featurelist
## ---------------------- Dataloaders ---------------------- ##
class H5Dataset(data.Dataset):

    def __init__(self, path,task,user):
        
        self.file_path = path
        self.dataset = None
        self.task=task
        
        
        self.distance={}
        







        with h5py.File(path, 'r') as f:
            self.dataset_len = (len(f["thermal"])-10)//15
        if self.task=='stress':
 
            self.dataset_len=self.dataset_len//4
   
            
        if self.task=='count' or self.task=='jelly' or self.task=='relax' :
            self.labels=torch.zeros(self.dataset_len)
        else:
            self.labels=torch.ones(self.dataset_len)
            #print(path)
        self.dataset =  None
        
        mapping={'brian': (6.12170518972332, 2.738155744105745), 'yuxuan': (2.0315562459903767, 1.5920591764902547), 'jingyu': (2.46085564398223, 2.6931856119445623), 'kirat': (0.7343852279005525, 0.3855071339384172), 'ruipeng': (6.327232446714744, 1.5537809487986232), 'sawin': (0.4034144615692554, 0.36409854590922397), 'huaiyu': (12.993516515114377, 2.2831871745269106), 'avery': (3.029731816350711, 0.9548403458802), 'coung': (4.185422265297203, 2.2880900195114644), 'yue': (3.671172082326892, 2.062053762316798),'vicki': (2.647892785887097, 1.969948917103175), 'moria': (7.3366887202380955, 1.8039903986669337), 'missy': (10.627276104098362, 3.302517839261801), 'jordan': (0.22407646935483871, 0.2712477562927061), 'gabrielle': (2.5076784529761906, 1.0701750774829135), 'shocky': (1.834749351428571, 0.8893038932418217), 'emily': (0.37484980344129554, 0.1071017964909624), 'carolyn': (5.16409214063745, 0.9667285651523427), 'joshua': (18.791469480708663, 5.734413946086138), 'lorn': (1.1227593221518988, 1.4207486203136652), 'timothy': (0.5828537324108818, 0.6048582465276751), 'yuhsun': (2.383287602291326, 0.6819557269910703), 'colleen': (1.3359034143243245, 1.3222589681695731), 'georgie': (0.417138202360515, 0.2145258540252646), 'hannah': (5.529630988802084, 0.8766097280926766), 'kaite': (13.0041588308, 2.8773194679990675),'kara': (0.15257421383363473, 0.10102582678159579), 'lohith': (0.1491865224022879, 0.07598298815996307), 'lydia': (0.166922578407225, 0.02978501354612667), 'shaily': (1.3505397171900162, 0.9783480415001867), 'susan': (0.49087909404315205, 0.3126269549023632), 'tabitha': (0.444800847970174, 0.07149893286787508),'christine': (6.680975991561182, 2.1038563665621)}
        user = ''.join([i for i in user if not i.isdigit()])
        self.user=user
        self.mean,self.std=mapping[user]
        far=['avery','brian','yuxuan','coung','ruipeng','yue','sawin']
        self.distance=0
        if user in far:
            self.distance=1


    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset =  h5py.File(self.file_path, 'r', libver="latest", swmr=True)
        
        label = (self.labels[index])
        label=label.type(torch.FloatTensor)
        #thermal=self.dataset["thermal"][index]
        #mask=self.dataset["mask"][index]
        if self.task=='stress':
            index+=self.dataset_len
        thermal=(self.dataset["thermal"][15*index:15*index+25].astype(np.float32))
        mask=(self.dataset["mask"][15*index:15*index+25].astype(np.float32))
        box=(self.dataset["z_box"][15*index:15*index+25].astype(np.float32))

        mask[mask>0]=1

        thermal=thermal*mask
        idxes=np.nonzero(mask)
        value=thermal[idxes]
        std=np.std(value, axis=0)
        mean=np.mean(value, axis=0)
        raw=thermal

        thermal=(thermal-mean)/std

        
        ppg=torch.from_numpy(self.dataset["ppg"][3*index:3*index+5])
        eda=(self.dataset["eda"][3*index:3*index+5])
        eda=(eda-self.mean)/(self.std)
        eda=EDAanalysis(eda)
        eda=torch.from_numpy(eda)
        
        
        #ppg=ppg.view(-1)
        std, mean=torch.std_mean(ppg,1)
        ppg=mean
        
        ppg=ppg.view(-1)
        

        return thermal ,label,eda,ppg,self.user#,self.distance


    def __len__(self):
        return self.dataset_len




def get_train_dataloader(  batch_size,idx):
    all_train_set = []
    all_test_set = []
    all_val_set = []
    #task_names = ['jelly', 'animal', 'stress',  'count', 'arithmetic', 'good', 'bad','prepareSong']
    task_names = ['arithmetic', 'stress', 'jelly','count']#

    #all_test=[['gabrielle','lohith7','moria','huaiyu','jingyu'],
    #['shocky','susan8','emily','yuxuan','vicki'],
    # ['lorn','carolyn','missy''kirat','shaily5'],
    #]
    
    all_val=[['georgie','kara8','hannah','yue','coung'],
    ['ruipeng','avery','lydia7','carolyn','jordan'],
    ['christine','yuhsun','tabitha5','brian','sawin'],
    ['shocky','susan8','emily','yuxuan','vicki'],
    ['lorn','carolyn','missy','kirat','shaily5'],
    ]
    val_name=all_val[idx]
    
    
    all_label=[]

    all_user=[]
    path = '../dessa_mask/'
    for f in os.listdir(path):
    
        if f[-1] != 'p':
            continue

        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        if user in val_name:
            continue

        all_user.append(user)
        t = H5Dataset(path + f, task,user.replace('3','').replace('2','').replace('1',''))
        all_train_set.append(t)
        all_label.append(t.labels)
    path = '../new_mask/'
    for f in os.listdir(path):
    
        if f[-1] != 'p':
            continue

        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        if user in val_name:
            continue

        all_user.append(user)
        t = H5Dataset(path + f, task,user.replace('3','').replace('2','').replace('1',''))
        all_train_set.append(t)
        all_label.append(t.labels)
        
    path = '../oct_mask/'
    for f in os.listdir(path):
    
        if f[-1] != 'p':
            continue

        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        if user in val_name:
            continue

        all_user.append(user)
        t = H5Dataset(path + f, task,user.replace('3','').replace('2','').replace('1',''))
        all_train_set.append(t)
        all_label.append(t.labels)

    all_label=torch.cat(all_label,0)
    weight_per_class=[0]*2
    weight_per_class[0] = len(all_label)/float(len(all_label)-all_label.sum()) 
    weight_per_class[1] = len(all_label)/float(all_label.sum()) 
    
    weight = [0] * len(all_label)                                              
    for idx, val in enumerate(all_label): 
        weight[idx] = weight_per_class[int(val.item())] 
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight)) 
    
    val_labels=[]
    path = '../dessa_mask/'
    for f in os.listdir(path):
    
        if f[-1] != 'p':
            continue

        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        if user not in val_name:
            continue

        t = H5Dataset(path + f, task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

        
    path='../new_mask/'
    for f in os.listdir(path):
        if f[-1] != 'p':
            continue
        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        
        if user not in val_name:
            continue

        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        t = H5Dataset(path + f,  task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

            
    path='../oct_mask/'
    for f in os.listdir(path):
        if f[-1] != 'p':
            continue
        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        
        if user not in val_name :
            continue

        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        t = H5Dataset(path + f,  task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)


    val_set = torch.utils.data.ConcatDataset(all_val_set)
    train_set = torch.utils.data.ConcatDataset(all_train_set)

    train_data_loader = data.DataLoader(dataset=train_set,
                                        num_workers=4,
                                        batch_size=batch_size,
                                        sampler=sampler,
                                       drop_last=True,
                                       
                                       )


    valid_data_loader = data.DataLoader(dataset=val_set,
                                        num_workers=4,
                                        batch_size=batch_size,
                                       drop_last=True,
                                       shuffle=False,
                                        
                                       
                                       
                                       )

    # convert labels -> category

    return train_data_loader, valid_data_loader



def get_test_dataloader(  batch_size,idx):
   
    all_test_set = []

    task_names = ['bad', 'prepareSong',   'jelly','count']#

    all_val=[['georgie','kara8','hannah','yue','coung'],
    ['ruipeng','avery','lydia7','carolyn','jordan'],
    ['christine','yuhsun','tabitha5','brian','sawin'],
    ['shocky','susan8','emily','yuxuan','vicki'],
    ['lorn','carolyn','missy','kirat','shaily5'],
    ]
    val_name=all_val[idx]
    
    all_val_set=[]
    val_labels=[]
    path = '../dessa_mask/'
    for f in os.listdir(path):
    
        if f[-1] != 'p':
            continue

        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        if user not in val_name:
            continue

        t = H5Dataset(path + f, task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

        
    path='../new_mask/'
    for f in os.listdir(path):
        if f[-1] != 'p':
            continue
        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        
        if user not in val_name:
            continue

        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        t = H5Dataset(path + f,  task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

            
    path='../oct_mask/'
    for f in os.listdir(path):
        if f[-1] != 'p':
            continue
        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        
        if user not in val_name :
            continue

        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        t = H5Dataset(path + f,  task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)


    val_set = torch.utils.data.ConcatDataset(all_val_set)



    valid_data_loader = data.DataLoader(dataset=val_set,
                                        num_workers=4,
                                        batch_size=batch_size,
                                       drop_last=True,
                                        
                                       
                                       
                                       )

    # convert labels -> category

    return valid_data_loader

def get_task_dataloader(  batch_size,idx,user,task):
   
    all_val_set = []
    all_label=[]
    all_user=[]
   
    task_names = [task]#
    val_name = [user]

    val_labels=[]
    path = '../dessa_mask/'
    for f in os.listdir(path):
    
        if f[-1] != 'p':
            continue

        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        if user not in val_name:
            continue

        t = H5Dataset(path + f, task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

        
    path='../new_mask/'
    for f in os.listdir(path):
        if f[-1] != 'p':
            continue
        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        
        if user not in val_name:
            continue

        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        t = H5Dataset(path + f,  task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

            
    path='../oct_mask/'
    for f in os.listdir(path):
        if f[-1] != 'p':
            continue
        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        
        if user not in val_name :
            continue

        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        t = H5Dataset(path + f,  task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

    test_set = torch.utils.data.ConcatDataset(all_val_set)


    data_loader = data.DataLoader(dataset=test_set,
                                        num_workers=4,
                                        batch_size=batch_size,
                                       drop_last=True,
                                        
                                       
                                       
                                       )

    # convert labels -> category

    return data_loader


def get_user_dataloader(  batch_size,idx,user):
   
    all_val_set = []
    all_label=[]
    all_user=[]
   
    task_names = ['arithmetic', 'stress', 'jelly','count']#
    val_name = [user]

    val_labels=[]
    path = '../dessa_mask/'
    for f in os.listdir(path):
    
        if f[-1] != 'p':
            continue

        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        if user not in val_name:
            continue

        t = H5Dataset(path + f, task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

        
    path='../new_mask/'
    for f in os.listdir(path):
        if f[-1] != 'p':
            continue
        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        
        if user not in val_name:
            continue

        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        t = H5Dataset(path + f,  task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

            
    path='../oct_mask/'
    for f in os.listdir(path):
        if f[-1] != 'p':
            continue
        f_split = f.split("_")
        user = f_split[-1][:-2]  # remove the '.p' suffix
        task = f_split[-2]
        
        if user not in val_name :
            continue

        if task not in task_names:
            continue
        if user[-1]=='3':
            continue
        t = H5Dataset(path + f,  task,user.replace('3','').replace('2','').replace('1',''))
        if user in val_name:
            all_val_set.append(t)

    test_set = torch.utils.data.ConcatDataset(all_val_set)


    data_loader = data.DataLoader(dataset=test_set,
                                        num_workers=4,
                                        batch_size=batch_size,
                                       drop_last=True,
                                        
                                       
                                       
                                       )

    # convert labels -> category

    return data_loader

