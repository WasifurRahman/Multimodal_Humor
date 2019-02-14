#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:06:55 2019

@author: echowdh2
"""


import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau

import h5py
import time
import data_loader as loader
from collections import defaultdict, OrderedDict
import argparse
import pickle as pickle
import time
import json, os, ast, h5py


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


class MFN(nn.Module):
    def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
        super(MFN, self).__init__()
        [self.d_l,self.d_a,self.d_v] = config["input_dims"]
        [self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
        total_h_dim = self.dh_l+self.dh_a+self.dh_v
        self.mem_dim = config["memsize"]
        window_dim = config["windowsize"]
        output_dim = 1
        attInShape = total_h_dim*window_dim
        gammaInShape = attInShape+self.mem_dim
        final_out = total_h_dim+self.mem_dim
        h_att1 = NN1Config["shapes"]
        h_att2 = NN2Config["shapes"]
        h_gamma1 = gamma1Config["shapes"]
        h_gamma2 = gamma2Config["shapes"]
        h_out = outConfig["shapes"]
        att1_dropout = NN1Config["drop"]
        att2_dropout = NN2Config["drop"]
        gamma1_dropout = gamma1Config["drop"]
        gamma2_dropout = gamma2Config["drop"]
        out_dropout = outConfig["drop"]

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)
        
        
    
    def forward(self,x):
        
        x_l = x[:,:,:self.d_l]
        x_a = x[:,:,self.d_l:self.d_l+self.d_a].unsqueeze(0)
        x_v = x[:,:,self.d_l+self.d_a:]
        # x is t x n x d
        n = x.shape[1]
        t = x.shape[0]
        self.h_l = torch.zeros(n, self.dh_l).cuda()
        self.h_a = torch.zeros(n, self.dh_a).cuda()
        self.h_v = torch.zeros(n, self.dh_v).cuda()
        self.c_l = torch.zeros(n, self.dh_l).cuda()
        self.c_a = torch.zeros(n, self.dh_a).cuda()
        self.c_v = torch.zeros(n, self.dh_v).cuda()
        self.mem = torch.zeros(n, self.mem_dim).cuda()
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_mems = []
        for i in range(t):
            # prev time step
            prev_c_l = self.c_l
            prev_c_a = self.c_a
            prev_c_v = self.c_v
            # curr time step
            new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
            # concatenate
            prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
            new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
            cStar = torch.cat([prev_cs,new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            attended = attention*cStar
            cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended,self.mem], dim=1)
            gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1*self.mem + gamma2*cHat
            all_mems.append(self.mem)
            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)

        # last hidden layer last_hs is n x h
        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        return output
        
 
class Unimodal_Context(nn.Module):
    def __init__(self,_config):
        super(Unimodal_Context, self).__init__()

        relevant_config = _config["unimodal_context"]
        #TODO: Must change it id text is sent as embedding. ANother way is to make the change in config file directly
        
        self.text_LSTM = nn.LSTM(input_size = relevant_config["text_lstm_input"],
                    hidden_size = relevant_config["hidden_size"],
                    batch_first=True)
        self.audio_LSTM = nn.LSTM(input_size = relevant_config["audio_lstm_input"],
                    hidden_size = relevant_config["hidden_size"],
                    batch_first=True)
        self.video_LSTM  = nn.LSTM(input_size = relevant_config["video_lstm_input"],
                    hidden_size = relevant_config["hidden_size"],
                    batch_first=True)
        self.device = _config["device"]
        self.hidden_size = relevant_config["hidden_size"]
        self.input_dims = _config["input_dims"]
        
    def forward(self,X_context):
        batch_size,context_size,seq_len,num_feats = X_context.size()

        X_context = torch.reshape(X_context,(batch_size*context_size,seq_len,num_feats)).to(self.device)
        
        new_batch_size = batch_size*context_size

        print("\nX_context:",X_context.size())
        
        text_context = X_context[:,:,:self.input_dims[0]]
        audio_context = X_context[:,:,self.input_dims[0]:self.input_dims[0]+self.input_dims[1]]
        video_context = X_context[:,:,self.input_dims[0]+self.input_dims[1]:]
        
        print("t:",text_context.shape,"a:",audio_context.shape,"v:",video_context.shape)

        
        #As LSTM accepts only (batch,seq_len,feats), we are reshaping the tensor.However,
        #it should not have any problem. There may be some issues during backprop, but lets see what happens
        
        #The text lstm
        ht_l = torch.zeros(new_batch_size, self.hidden_size).unsqueeze(0).to(self.device)
        ct_l = torch.zeros(new_batch_size, self.hidden_size).unsqueeze(0).to(self.device)
        _,(ht_last,ct_last) = self.text_LSTM(text_context,(ht_l,ct_l))
        print(ht_last.shape)
        
        
        ha_l = torch.zeros(new_batch_size, self.hidden_size).unsqueeze(0).to(self.device)
        ca_l = torch.zeros(new_batch_size, self.hidden_size).unsqueeze(0).to(self.device)
        _,(ha_last,ca_last) = self.audio_LSTM(audio_context,(ha_l,ca_l))
        print(ha_last.shape)
        
        hv_l = torch.zeros(new_batch_size, self.hidden_size).unsqueeze(0).to(self.device)
        cv_l = torch.zeros(new_batch_size, self.hidden_size).unsqueeze(0).to(self.device)
        _,(hv_last,cv_last) = self.audio_LSTM(audio_context,(hv_l,cv_l))
        print(hv_last.shape)
        
        text_lstm_result = torch.reshape(ht_last,(batch_size,context_size,-1))
        audio_lstm_result = torch.reshape(ha_last,(batch_size,context_size,-1))
        video_lstm_result = torch.reshape(hv_last,(batch_size,context_size,-1))
        print("final reslt from unimodal:",text_lstm_result.shape,audio_lstm_result.shape,video_lstm_result.shape)

        
        return text_lstm_result,audio_lstm_result,video_lstm_result
        
        

class Contextual_MFN(nn.Module):
    def __init__(self,_config):
        super(Contextual_MFN, self).__init__()

        self.unimodal_context = Unimodal_Context(_config)
    def forward(self,X_Punchline,X_Context,Y):
        text_uni,audio_uni,video_uni = self.unimodal_context.forward(X_Context)
        
        
        
        
