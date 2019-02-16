#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 07:59:01 2019

@author: echowdh2
"""

import faulthandler
faulthandler.enable()
import sys
import numpy as np
import random
import torch
import tqdm
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

import h5py
import time
from collections import defaultdict, OrderedDict
import argparse
import pickle
import time
import json, os, ast, h5py
import math

from models import MFN
from models import Contextual_MFN


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from sacred import Experiment
from tqdm import tqdm
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

ex = Experiment('multimodal_humor')
from sacred.observers import MongoObserver

#We must change url to the the bluehive node on which the mongo server is running
url_database = 'bhc0086:27017'
mongo_database_name = 'prototype'
ex.observers.append(MongoObserver.create(url= url_database ,db_name= mongo_database_name))

@ex.config
def cfg():
    node_index = 0
    epoch = 75 #paul did 50
    shuffle = True
    num_workers = 2
    best_model_path =  "/scratch/echowdh2/saved_models_from_projects/multimodal_transformer/"+str(node_index) +"_best_model.chkpt"
    num_context_sequence=5
    
    dataset_location = None
    dataset_name = None
    text_indices = None
    audio_indices=None
    video_indices = None
    max_seq_len = None
    input_dims=None #organized as [t,a,v]
    
    padding_value = 0.0
    
    #To ensure that it captures the whole batch at the same time
    #and hence we get same score as Paul
    #TODO: Must cahange
    train_batch_size = random.choice([64,128,256,512])
    #These two are coming from running_different_configs.py
    dev_batch_size=None
    test_batch_size=None

   
    
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    save_model = "best_model"
    save_mode = 'best'
    
    prototype=False
    if prototype:
        epoch=1
        
    #TODO: May have to change the hidden_sizes to match with later stages
    #TODO:Will need to add RANDOM CHOICE FOR hidden_size later
    #Basically the hidden_sizes is an arry containing hidden_size for all three [t,a,v]    
    unimodal_context = {"text_lstm_input":input_dims[0],"audio_lstm_input":input_dims[1],
                        "video_lstm_input":input_dims[2],"hidden_sizes":[64,8,16]
                                                     }
    
    multimodal_context_configs = {'d_word_vec':512,'d_model':512,'d_inner_hid':2048,
                   'd_k':64,'d_v':64,'n_head':8,'n_layers':6,'n_warmup_steps':4000,
                   'dropout':0.1,'embs_share_weight':True,'proj_share_weight':True,
                   'label_smoothing': True,'max_token_seq_len':num_context_sequence,
                   'n_source_features':sum(unimodal_context["hidden_sizes"]),
                   'post_encoder':{'mfn_mem_input_drop':random.choice([0.0,0.2,0.5,0.7])}
                   
                   }

        
    #All these are mfn configs    
    config = dict()
    config["input_dims"] = input_dims
    hl = random.choice([32,64,88,128,156,256])
    ha = random.choice([8,16,32,48,64,80])
    hv = random.choice([8,16,32,48,64,80])
    config["h_dims"] = [hl,ha,hv]
    config["memsize"] = random.choice([64,128,256,300,400])
    config["windowsize"] = 2
    config["batchsize"] = random.choice([32,64,128,256])
    config["num_epochs"] = 50
    config["lr"] = random.choice([0.001,0.002,0.005,0.008,0.01])
    config["momentum"] = random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
    
    NN1Config = dict()
    NN1Config["shapes"] = random.choice([32,64,128,256])
    NN1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    NN2Config = dict()
    NN2Config["shapes"] = random.choice([32,64,128,256])
    NN2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    gamma1Config = dict()
    gamma1Config["shapes"] = random.choice([32,64,128,256])
    gamma1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    gamma2Config = dict()
    gamma2Config["shapes"] = random.choice([32,64,128,256])
    gamma2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    outConfig = dict()
    outConfig["shapes"] = random.choice([32,64,128,256])
    outConfig["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    mfn_configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
   


class Generic_Dataset(Dataset):
    def __init__(self, X, Y,_config):
        self.X = X
        self.Y = Y
        self.config = _config
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        
        #print("The P:",X.size(),X[:,:])
        #TODO: Must change it when correct dataset arrives
        #we are just repeating each entry 
        X_context = torch.FloatTensor(np.repeat(self.X[idx],
            self.config["num_context_sequence"],0).reshape((self.config["num_context_sequence"],
                       self.config["max_seq_len"],-1))) 
        #print("The Context:",X_context.size())
        
        #Basically, we will think the whole sentence as a sequence.
        #all the words will be merged. If all of them are zero, then it is a padding 
        reshaped_context = torch.reshape(X_context,(X_context.shape[0],-1))
        #print("The reshaped context:",reshaped_context.size())
        padding_rows = np.where(~reshaped_context.cpu().numpy().any(axis=1))[0]
        n_rem_entries= reshaped_context.shape[0] - len(padding_rows)
        X_pos_context = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
        #print("X_pos:",X_pos," Len:",X_pos.shape)
        X_pos_context = torch.LongTensor(X_pos_context)   
        #print("X_pos_context:",X_pos_context.shape,X_pos_context)

        Y = torch.FloatTensor([self.Y[idx]])
        return X,X_context,X_pos_context,Y




@ex.capture        
def load_saved_data(_config):
    
    data_path = os.path.join(_config["dataset_location"],'data')
    #TODO:Change it properly
    
    h5f = h5py.File(os.path.join(data_path,'X_train.h5'),'r')
    X_train = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'y_train.h5'),'r')
    y_train = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'X_valid.h5'),'r')
    X_valid = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'y_valid.h5'),'r')
    y_valid = h5f['data'][:]
    h5f.close()
    
    h5f = h5py.File(os.path.join(data_path,'X_test.h5'),'r')
    X_test = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'y_test.h5'),'r')
    y_test = h5f['data'][:]
    h5f.close()
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

@ex.capture
def set_up_data_loader(_config):
    train_X,train_Y,dev_X,dev_Y,test_X,test_Y = load_saved_data()
    print("all data loaded. Now creating data loader")
    
    if(_config["prototype"]):
        train_X = train_X[:10,:,:]
        train_Y = train_Y[:10]
        
        dev_X = dev_X[:10,:,:]
        dev_Y = dev_Y[:10]
        
        test_X = test_X[:10,:,:]
        test_Y = test_Y[:10]
        
        
    train_dataset = Generic_Dataset(train_X,train_Y,_config = _config)
    dev_dataset = Generic_Dataset(dev_X,dev_Y,_config=_config)
    test_dataset = Generic_Dataset(test_X,test_Y,_config=_config)
    
    train_dataloader = DataLoader(train_dataset, batch_size=_config["train_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    dev_dataloader = DataLoader(dev_dataset, batch_size=_config["dev_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    test_dataloader = DataLoader(test_dataset, batch_size=_config["test_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    
    #print(train_X.shape,train_Y.shape,dev_X.shape,dev_Y.shape,test_X.shape,test_Y.shape)
    #data_loader = test_data_loader(train_X,train_Y,_config)
    return train_dataloader,dev_dataloader,test_dataloader

@ex.capture
def set_random_seed(_seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)

@ex.capture
def train_mfn(train_data_loader,valid_data_loader,test_data_loader,_config):
    print(_config)
    
    #In our program, we through it as (batch_size,max_seq_len_num_features).
    #For doing that, LSTM has to programeed as batch_first=true manner. SO, we need to swap axes.
    #Now, paul did not use any dataloader. So, he does it here. We can do it in the train loop
    #for each batch
#     X_train = X_train.swapaxes(0,1)
#     X_valid = X_valid.swapaxes(0,1)
#     X_test = X_test.swapaxes(0,1)
    #torch.permute(1,0,2) is our solution
    
    [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = _config["mfn_configs"]

    model = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

    optimizer = optim.Adam(model.parameters(),lr=config["lr"])

    criterion = nn.L1Loss()

    model = model.to(_config["device"])
    criterion = criterion.to(_config["device"])
    scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)
    

@ex.capture
def train_epoch(model, training_data, criterion,optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    epoch_loss = 0.0
    num_batches = 0
   
    for batch in tqdm(training_data, mininterval=2,desc='  - (Training)   ', leave=False):

     #TODO: For simplicity, we are not using X_pos right now as we really do not know
     #how it can be used properly. So, we will just use the context information only.
        X_Punchline,X_Context,X_pos_Context,Y = map(lambda x: x.to(device), batch)
        
        print("\nData_size:\nX_P:", X_Punchline.shape,", X_C:",X_Context.shape," Y:",Y.shape)
        #TODO: Doing it to avoid error. Must remove it afterwards.
        predictions = model(X_Punchline,X_Context,X_pos_Context,Y)
        # forward
# =============================================================================
#         optimizer.zero_grad()
#         predictions = model(train_X,train_X_Context,train_Y).squeeze(0)
#         #print(predictions.size(),train_Y.size())
# 
#         loss = criterion(predictions, train_Y)
#         loss.backward()
#         #optimizer.step()
#         epoch_loss += loss.item()
# 
#         # update parameters
#         optimizer.step_and_update_lr()
#         
#         num_batches +=1
# =============================================================================
    #TODO: MUST REMOVE
    if(num_batches==0):
        num_batches+=1
    return epoch_loss / num_batches

@ex.capture
def eval_epoch(model,data_loader,criterion, device):
    ''' Epoch operation in evaluation phase '''
    epoch_loss = 0.0
    num_batches=0
    model.eval()
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Validation)   ', leave=False):
    
         
            X,X_pos,Y = map(lambda x: x.to(device), batch)
            predictions = model(X,X_pos,Y).squeeze(0)
            loss = criterion(predictions, Y)
            
            epoch_loss += loss.item()
            
            num_batches +=1
    return epoch_loss / num_batches
@ex.capture
def reload_model_from_file(file_path):
        checkpoint = torch.load(file_path)
        _config = checkpoint['_config']
        
        encoder_config = _config["encoder"]
        model = Transformer(
        
        n_src_features = encoder_config["n_source_features"],
        len_max_seq = encoder_config["max_token_seq_len"],
        _config = _config,
        tgt_emb_prj_weight_sharing=encoder_config["proj_share_weight"],
        emb_src_tgt_weight_sharing=encoder_config["embs_share_weight"],
        d_k=encoder_config["d_k"],
        d_v=encoder_config["d_v"],
        d_model=encoder_config["d_model"],
        d_word_vec=encoder_config["d_word_vec"],
        d_inner=encoder_config["d_inner_hid"],
        n_layers=encoder_config["n_layers"],
        n_head=encoder_config["n_head"],
        dropout=encoder_config["dropout"]
        ).to(_config["device"])
        

        

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        return model
        
@ex.capture
def test_epoch(model,data_loader,criterion, device):
    ''' Epoch operation in evaluation phase '''
    epoch_loss = 0.0
    num_batches=0
    model.eval()
    returned_Y = None
    returned_predictions = None
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Validation)   ', leave=False):
    
         
            X,X_pos,Y = map(lambda x: x.to(device), batch)
            predictions = model(X,X_pos,Y).squeeze(0)
            loss = criterion(predictions, Y)
            
            epoch_loss += loss.item()
            
            num_batches +=1
            #if we don'e do the squeeze, it remains as 2d numpy arraya nd hence
            #creates problems like nan while computing various statistics on them
            returned_Y = Y.squeeze(1).cpu().numpy()
            returned_predictions = predictions.squeeze(1).cpu().data.numpy()

    return returned_predictions,returned_Y   


    
            
@ex.capture
def train(model, training_data, validation_data, optimizer,criterion,_config,_run):
    ''' Start training '''
    model_path = _config["best_model_path"]

    valid_losses = []
    for epoch_i in range(_config["epoch"]):
        
        train_loss = train_epoch(
            model, training_data, criterion,optimizer, device = _config["device"],
                smoothing=_config["multimodal_context_configs"]["label_smoothing"])
    #     #print("\nepoch:{},train_loss:{}".format(epoch_i,train_loss))
    #     _run.log_scalar("training.loss", train_loss, epoch_i)


    #     valid_loss = eval_epoch(model, validation_data, criterion,device=_config["device"])
    #     _run.log_scalar("dev.loss", valid_loss, epoch_i)
        
    #     #scheduler.step(valid_loss)

        
        
    #     valid_losses.append(valid_loss)
    #     #print("\nepoch:{},train_loss:{}, valid_loss:{}".format(epoch_i,train_loss,valid_loss))

    #     model_state_dict = model.state_dict()
    #     checkpoint = {
    #         'model': model_state_dict,
    #         '_config': _config,
    #         'epoch': epoch_i}

    #     if _config["save_model"]:
    #         if _config["save_mode"] == 'best':
    #             if valid_loss <= min(valid_losses):
    #                 torch.save(checkpoint, model_path)
    #                 print('    - [Info] The checkpoint file has been updated.')
    # #After the entire training is over, save the best model as artifact in the mongodb, only if it is not protptype
    # if(_config["protptype"]==False):
    #     ex.add_artifact(model_path)


@ex.capture
def test_score(test_data_loader,criterion,_config,_run):
    model_path =  _config["best_model_path"]
    model = reload_model_from_file(model_path)

    predictions,y_test = test_epoch(model,test_data_loader,criterion,_config["device"])
    #print("predictions:",predictions,predictions.shape)
    #print("ytest:",y_test,y_test.shape)
    mae = np.mean(np.absolute(predictions-y_test))
    print("mae: ", mae)
    
    corr = np.corrcoef(predictions,y_test)[0][1]
    print("corr: ", corr)
    
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    print("mult_acc: ", mult)
    
    f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
    print("mult f_score: ", f_score)
    
    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)
    print("Confusion Matrix :")
    confusion_matrix_result = confusion_matrix(true_label, predicted_label)
    print(confusion_matrix_result)
    
    print("Classification Report :")
    classification_report_score = classification_report(true_label, predicted_label, digits=5)
    print(classification_report_score)
    
    accuracy = accuracy_score(true_label, predicted_label)
    print("Accuracy ",accuracy )
    
    _run.info['final_result']={'accuracy':accuracy,'mae':mae,'corr':corr,"mult_acc":mult,
             "mult_f_score":f_score,"Confusion Matrix":confusion_matrix_result,
             "Classification Report":classification_report_score}
    return accuracy



@ex.automain
def driver(_config):
    
    set_random_seed()
    #print("inside driver")
    #X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data()
    #print(X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_data_loader,dev_data_loader,test_data_loader = set_up_data_loader()
    
    multimodal_context_config = _config["multimodal_context_configs"]
    
    model = Contextual_MFN(_config).to(_config["device"])
    #for now, we will use the same scheduler for the entire model.
    #Later, if necessary, we may use the default optimizer of MFN
    #TODO: May have to use separate scheduler for transformer and mfn
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        multimodal_context_config["d_model"], multimodal_context_config["n_warmup_steps"])
    
    #TODO: May have to change the criterion
    criterion = nn.L1Loss()
    criterion = criterion.to(_config["device"])
    
    # optimizer =  optim.Adam(
    #         filter(lambda x: x.requires_grad, transformer.parameters()),lr = _config["learning_rate"],
    #         betas=(0.9, 0.98), eps=1e-09)
    #torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=False)
    #scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)

    train(model, train_data_loader,dev_data_loader, optimizer,criterion)
    
# =============================================================================
#     test_accuracy = test_score(test_data_loader,criterion)
#     ex.log_scalar("test.accuracy",test_accuracy)
#     results = dict()
#     #I believe that it will try to minimize the rest. Let's see how it plays out
#     results["optimization_target"] = 1 - test_accuracy
# 
#     return results
# =============================================================================

