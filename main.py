import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import numpy as np
import os
import optuna
import pickle
import sys
import random
from function import *
import models
from torch.optim.lr_scheduler import ExponentialLR 
from sklearn import metrics
def setseed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.use_deterministic_algorithms(True)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

class Engine:
    def __init__(self,model,optimizer,device,idx,filename):
        self.model=model
        self.device=device
        self.optimizer=optimizer
        self.idx=str(idx)
        self.filename=filename
        self.loss_recon = nn.MSELoss()
        self.diver=nn.KLDivLoss(reduction='batchmean')

        
    @staticmethod 
    def loss_fn(outputs,target):
        Loss=nn.CrossEntropyLoss()
        return Loss(outputs,target)
    
    def train(self,data_loader):
        self.model.train()
        final_loss = 0
        truth=[]
        predict=[]
        for batch_idx, (data,label,eda,ppg,name) in enumerate(data_loader):

            data,eda,label = data.to(self.device).float(), eda.to(self.device).float(),label.to(self.device).long()
            data, eda ,label= Variable(data), Variable(eda),Variable(label)

            outputs = self.model(data, eda)
            loss,y_hat,logit = self.model.get_loss(outputs, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            final_loss+= loss.item()
            truth.extend(label.tolist())
            predict.extend(y_hat.tolist())

        trainacc=accuracy_score(truth,predict)
        cf = confusion_matrix(truth,predict)
        f1score=f1_score(truth,predict,average='weighted')
        with open("./"+self.filename+'/logs_'+self.idx+'.txt', 'a+') as f:
            print('Train : Loss: {:.4f}, Train acc : {:.4f}, Train f1 : {:.4f}'.format(final_loss/len(data_loader),trainacc,f1score),file=f)
        return f1score
    
    def evalue(self,data_loader):
        self.model.eval()
        final_loss = 0
        truth=[]
        predict=[]
        nameid=[]
        raw=[]
        for batch_idx, (data,label,eda,ppg,name) in enumerate(data_loader):
            data,eda,label = data.to(self.device).float(), eda.to(self.device).float(),label.to(self.device).long()
            data, eda ,label= Variable(data), Variable(eda),Variable(label)
            

            outputs = self.model.infer(data, eda)
            loss,y_hat,logit = self.model.get_loss(outputs, label)
            
            final_loss+= loss.item()
            truth.extend(label.tolist())
            predict.extend(y_hat.tolist())
            raw.extend(logit)
            nameid.extend(name)
        uniq_id=np.unique(nameid)
        with open("./"+self.filename+'/logs_'+self.idx+'.txt', 'a+') as f:
            print('User wise',file=f,end =" ")

        for ids in uniq_id:
            table = zip(nameid, truth,predict)

            filt=list(filter(lambda s:s[0]==ids ,table))
            
            acc=accuracy_score(list(zip(*filt))[1],list(zip(*filt))[2])
            with open("./"+self.filename+'/logs_'+self.idx+'.txt', 'a+') as f:
                print('{:.4f}'.format(acc),file=f,end =",")
            
        valacc=accuracy_score(truth,predict)
        cf = confusion_matrix(truth,predict)
        auc=metrics.roc_auc_score(truth, raw, average='micro')
        f1score=f1_score(truth,predict,average='weighted')
        with open("./"+self.filename+'/logs_'+self.idx+'.txt', 'a+') as f:
            print('Val : Loss: {:.4f}, Val acc : {:.4f}, Val f1 : {:.4f}, Val AUC : {:.4f}'.format(final_loss/len(data_loader),valacc,f1score,auc),file=f)
            print(cf,file=f)
        return f1score

def objective(modelname,dataidx,trial=None,filename=None):
    
    

    params={
            
            "nchannels":16,
            "hidden_size":128,
            "hyper_size":32,
            "LR":trial.suggest_loguniform("LR",1e-5,1e-3),

            #"LR":trial.suggest_loguniform("LR",1e-6,1e-3),
            "dropout":0.2,
            "epoch":100,
   
    }
    
    setseed(2023)

    
    with open(filename+"params_trial_number_{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(params, fout)
    model_train = models.Models[modelname](hyper_size=params["hyper_size"],hidden_size=params["hidden_size"],dropout=params["dropout"],nchannels=params["nchannels"])
    
    #model_path='./'+'mm'+'/train'+'2'+'seed'+'1'+'.pth'
    #model_train.load_state_dict(torch.load(model_path))
    
    for name, layer in model_train.named_children():
        for n, l in layer.named_modules():
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()

    
    device_train = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(filename+'/logs_'+str(dataidx)+'.txt', 'a+') as f:
        print(device_train,file=f)
        print(torch.cuda.device_count(),file=f)
        print(torch.cuda.get_device_name(0),file=f)

    model_train.to(device_train)

    optimizer_train = optim.Adam(model_train.parameters(), lr=params["LR"])#
    #scheduler = ExponentialLR(optimizer_train, gamma=0.5)
    

    epoch=params["epoch"]
    eng =Engine(model_train,optimizer_train,device_train,dataidx,filename)
    best_f1=0
    best_test_f1=0
    early_stopping_itr=20
    e_s_counter=0

    for e in range(epoch):
        if(e==0):
            with open(filename+'/logs_'+str(dataidx)+'.txt', 'a+') as f:
                print("\n----------------------------------------NEW_TRIAL--------------------------------------\n",file=f)
                print('trail',trial.number,file=f)
        with open(filename+'/logs_'+str(dataidx)+'.txt', 'a+') as f:
            print("EPOCH: ",e,file=f)
        train_f1=eng.train(train_data_loader)
        valid_f1=eng.evalue(val_data_loader)

        if valid_f1 >= best_f1 and train_f1>0.7:
            best_f1 = valid_f1

           
            best_state=model_train.state_dict()
            torch.save(best_state, filename+"train"+str(trial.number)+'seed'+str(dataidx)+".pth")
        elif train_f1>0.7:
            e_s_counter+=1
            #scheduler.step()
        if(e_s_counter>early_stopping_itr):
            break

    return best_f1   

if __name__ == '__main__':

    paras=sys.argv[1:]
    data_idx=int(paras[0])
    modelname=paras[1]
    global train_data_loader,val_data_loader
    train_data_loader,val_data_loader=get_train_dataloader(batch_size=5,idx=data_idx)

    model_path='./'+modelname+'/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(model_path+'/logs_'+str(data_idx)+'.txt', 'w') as f:
        print("prgram start",file=f)
    
    sampler = optuna.samplers.TPESampler(seed=2023)
    study=optuna.create_study(sampler=sampler,direction="maximize",study_name="hyper")
    study.optimize(lambda trial: objective(modelname,data_idx,trial,model_path),n_trials=5)

    trial_=study.best_trial