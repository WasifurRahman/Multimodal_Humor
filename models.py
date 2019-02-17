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

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


class MFN(nn.Module):
    def __init__(self,_config):
        
        super(MFN, self).__init__()
        config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig = \
            _config["mfn_configs"]
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
        
        
    
    def forward(self,x,h_l_prior,h_a_prior,h_v_prior,mem_prior):
        
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
        #print("Unimodal configs:",relevant_config)
        #TODO: Must change it id text is sent as embedding. ANother way is to make the change in config file directly
        [self.h_text,self.h_audio,self.h_video] = relevant_config["hidden_sizes"]
        self.text_LSTM = nn.LSTM(input_size = relevant_config["text_lstm_input"],
                    hidden_size = self.h_text,
                    batch_first=True)
        self.audio_LSTM = nn.LSTM(input_size = relevant_config["audio_lstm_input"],
                    hidden_size = self.h_audio,
                    batch_first=True)
        self.video_LSTM  = nn.LSTM(input_size = relevant_config["video_lstm_input"],
                    hidden_size = self.h_video,
                    batch_first=True)
        self.device = _config["device"]
        #self.hidden_size = relevant_config["hidden_size"]
        self.input_dims = _config["input_dims"]
        
    def forward(self,X_context):
        old_batch_size,context_size,seq_len,num_feats = X_context.size()
        
        # #As LSTM accepts only (batch,seq_len,feats), we are reshaping the tensor.However,
        # #it should not have any problem. There may be some issues during backprop, but lets see what happens
        
        X_context = torch.reshape(X_context,(old_batch_size*context_size,seq_len,num_feats)).to(self.device)
        
        new_batch_size = old_batch_size*context_size

        #print("\nX_context:",X_context.size())
        
        text_context = X_context[:,:,:self.input_dims[0]]
        audio_context = X_context[:,:,self.input_dims[0]:self.input_dims[0]+self.input_dims[1]]
        video_context = X_context[:,:,self.input_dims[0]+self.input_dims[1]:]
        
        print("Context shapes:\n","t:",text_context.shape,"a:",audio_context.shape,"v:",video_context.shape)

        
       
        #The text lstm
        ht_l = torch.zeros(new_batch_size, self.h_text).unsqueeze(0).to(self.device)
        ct_l = torch.zeros(new_batch_size, self.h_text).unsqueeze(0).to(self.device)
        _,(ht_last,ct_last) = self.text_LSTM(text_context,(ht_l,ct_l))
        #print("ht_last:",ht_last.shape)
        
        
        ha_l = torch.zeros(new_batch_size, self.h_audio).unsqueeze(0).to(self.device)
        ca_l = torch.zeros(new_batch_size, self.h_audio).unsqueeze(0).to(self.device)
        _,(ha_last,ca_last) = self.audio_LSTM(audio_context,(ha_l,ca_l))
        #print("ha_last:",ha_last.shape)
        
        hv_l = torch.zeros(new_batch_size, self.h_video).unsqueeze(0).to(self.device)
        cv_l = torch.zeros(new_batch_size, self.h_video).unsqueeze(0).to(self.device)
        _,(hv_last,cv_last) = self.video_LSTM(video_context,(hv_l,cv_l))
        #print("ha last:",hv_last.shape)
        
        text_lstm_result = torch.reshape(ht_last,(old_batch_size,context_size,-1))
        audio_lstm_result = torch.reshape(ha_last,(old_batch_size,context_size,-1))
        video_lstm_result = torch.reshape(hv_last,(old_batch_size,context_size,-1))
        #print("final result from unimodal:",text_lstm_result.shape,audio_lstm_result.shape,video_lstm_result.shape)

        
        return text_lstm_result,audio_lstm_result,video_lstm_result
        


class Multimodal_Context(nn.Module):
    def __init__(self,_config):
        super(Multimodal_Context, self).__init__()
        print("Config in multimodal context:",_config["multimodal_context_configs"])
        self.config = _config
        (in_text,in_audio,in_video) =  [ _config["num_context_sequence"]*e for e in _config["unimodal_context"]["hidden_sizes"]]
        
        #mfn config contains a list of configs and the first one of them is the config, which
        #contains a dictionary called h_dims which has the [ht,ha,hv].
        (out_text,out_audio,out_video) = _config["mfn_configs"][0]["h_dims"]
        
        #The first one is hl
        self.fc_uni_text_to_mfn_text_input = nn.Linear(in_text,out_text)
        
        #The second one is ha
        self.fc_uni_audio_to_mfn_audio_input = nn.Linear(in_audio,out_audio)
        
        #The third one is hv
        self.fc_uni_video_to_mfn_video_input = nn.Linear(in_video,out_video)
        
        #This one will output the initialization of the mfn meory
        encoder_config =self.config["multimodal_context_configs"]
        self.self_attention_module = Transformer(
        
        n_src_features = encoder_config["n_source_features"],
        len_max_seq = encoder_config["max_token_seq_len"],
        _config = self.config,
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
        ).to(self.config["device"])

    
    def forward(self,text_uni,audio_uni,video_uni,X_pos_Context,Y):
        #So, we are getting three tensor corresponding to three modalities, each of shape:torch.Size([10, 5, 64])
        
        #We will initialize the text lstm of mfn solely from the result of text_uni.
        
        #Text_uni has shape [10,5,64], we will convert need to convert it to [batch_size,hidden_size].\
        #So, first, we can just convert it to [10,5*64] here 10 is the batch size.
        #The same is done with audio and video uni
        reshaped_text_uni = text_uni.reshape((text_uni.shape[0],-1))
        #print("reshaped text:",reshaped_text_uni.shape)
        reshaped_audio_uni = audio_uni.reshape((audio_uni.shape[0],-1))
        #print("reshaped audio:",reshaped_audio_uni.shape)
        reshaped_video_uni = video_uni.reshape((video_uni.shape[0],-1))
        #print("reshaped video:",reshaped_video_uni.shape)
        
        #Then, we will have three linear trans. So, all three reshaped tensors begin with 
        #shape (batch_size,config.num_context_sequence*config.unimodal_context.hidden_size) 
        #And we need to convert them to (batch_size,mfn_configs.config.[hl or ht or hv])
        #ht means hidden text
        #TODO: May use a dropout layer later
        mfn_hl_input = self.fc_uni_text_to_mfn_text_input(reshaped_text_uni)
        #ha means hidden audio
        mfn_ha_input = self.fc_uni_audio_to_mfn_audio_input(reshaped_audio_uni)
        #hv means hidden video
        mfn_hv_input = self.fc_uni_video_to_mfn_video_input(reshaped_video_uni)
        #These three will be used to initialize the three unimodal lstms of mfn
        #print("mfn text lstm hidden init:",mfn_ht_input.shape)
        #print("mfn audio lstm hidden init:",mfn_ha_input.shape)
        #print("mfn video lstm hidden init:",mfn_hv_input.shape)
        
        
        #Now, we will do self attention to convert all three original text_uni,audio_uni and video_uni 
        #to feed into transformer. They are of shape (10,5,64), (10,5,8) and (10,5,16). SO, we need to first concat them 
        #to convert them to shape (20,5,64+8+16=88). So, we will concat them by axis=2
        all_three_orig_concat = torch.cat([text_uni,audio_uni,video_uni],dim=2)
        print("all mods concatenated:",all_three_orig_concat.size())
        
        #Then, we are passing it through transformer
        mfn_mem_lstm_input = self.self_attention_module(all_three_orig_concat,X_pos_Context,Y).squeeze(0)
        
        #print("Getting output from transformer:",mfn_mem_lstm_input.size())
        
        return mfn_hl_input,mfn_ha_input,mfn_hv_input,mfn_mem_lstm_input
        
        
        
        
        
        

        

class Contextual_MFN(nn.Module):
    def __init__(self,_config):
        super(Contextual_MFN, self).__init__()
        
        #print("config in mfn)
        print("the config in mfn_configs:",_config["mfn_configs"][0])
        self.unimodal_context = Unimodal_Context(_config)
        self.multimodal_context = Multimodal_Context(_config)
        self.mfn = MFN(_config)
        
        
        
    def forward(self,X_Punchline,X_Context,X_pos_Context,Y):
        text_uni,audio_uni,video_uni = self.unimodal_context.forward(X_Context)
        print("unimodal complete:",text_uni.shape, audio_uni.shape, video_uni.shape)

        mfn_hl_input,mfn_ha_input,mfn_hv_input,mfn_mem_lstm_input = \
          self.multimodal_context.forward(text_uni,audio_uni,video_uni,X_pos_Context,Y)
          
        print("Ready to init the mfn with this:","L:",mfn_hl_input.shape,"A:",mfn_ha_input.shape,\
              "V:",mfn_hv_input.shape,"mem:",mfn_mem_lstm_input.shape)  
        
        
       
        

        
        
        
        
