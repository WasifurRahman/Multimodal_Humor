#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 08:04:26 2019

@author: echowdh2
"""

from driver import ex
import os
import argparse, sys
parser=argparse.ArgumentParser()
parser.add_argument('--dataset', help='the dataset you want to work on')

dataset_specific_config = {
        #Train:10569,dev:2642,Test:3303
        "TED_humor":{'input_dims':[1,81,75,300],'max_seq_len':20,'dev_batch_size':2645,'test_batch_size':3305},

        "mosi":{'input_dims':[300,5,20],'text_indices':(0,300),'audio_indices':(300,305),'video_indices':(305,325),'max_seq_len':20,'dev_batch_size':250,'test_batch_size':700},
        "iemocap":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "mmmo":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "moud":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "pom":{'text_indices':(0,300),'audio_indices':(300,343),'video_indices':(343,386),'max_seq_len':21},
        "youtube":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21}
        
        }
 # use_context=True
 #    use_context_text=True
 #    use_context_audio=True
 #    use_context_video = True
    
 #    use_punchline_text=True
 #    use_punchline_audio=True
 #    use_punchline_video=True
experiment_configs=[
        {'use_context':True,'use_punchline_text':True,'use_punchline_audio':True,'use_punchline_video':True},
        {'use_context':True,'use_punchline_text':True,'use_punchline_audio':False,'use_punchline_video':False},
        {'use_context':True,'use_punchline_text':True,'use_punchline_audio':True,'use_punchline_video':False},
        {'use_context':True,'use_punchline_text':True,'use_punchline_audio':False,'use_punchline_video':True},
        {'use_context':False,'use_punchline_text':True,'use_punchline_audio':True,'use_punchline_video':True},
        {'use_context':False,'use_punchline_text':True,'use_punchline_audio':False,'use_punchline_video':False},
        {'use_context':False,'use_punchline_text':True,'use_punchline_audio':True,'use_punchline_video':False},
        {'use_context':False,'use_punchline_text':True,'use_punchline_audio':False,'use_punchline_video':True}
        
        ]
num_experiments = len(experiment_configs)

#sacred will generate a different random _seed for every experiment
#and we will use that seed to control the randomness emanating from our libraries
node_index=int(os.environ['SLURM_ARRAY_TASK_ID'])
#node_index=0

#So, we are assuming that there will a folder called /processed_multimodal_data in the parent folder
#of this code. I wanted to keep it inside the .git folder. But git push limits file size to be <=100MB
#and some data files exceeds that size.
all_datasets_location = "../processed_multimodal_data"
def run_configs(dataset_location):
    #print(dataset_location)
    dataset_name = dataset_location[dataset_location.rfind("/")+1:]
    appropriate_config_dict = {**dataset_specific_config[dataset_name],**experiment_configs[node_index%num_experiments],"node_index":node_index,
                              "prototype":False,'dataset_location':dataset_location,"dataset_name":dataset_name}
    #print(appropriate_config_dict)
    #Just run it many times
    while(True):
        r= ex.run(config_updates=appropriate_config_dict)
    #r = ex.run(named_configs=['search_space'],config_updates={"node_index":node_index,"prototype":True})
    
    
#run it like ./running_different_configs.py --dataset=mosi
if __name__ == '__main__':
    args = parser.parse_args()
    dataset_path = os.path.join(all_datasets_location,args.dataset)
    if(os.path.isdir(dataset_path)):
        
        run_configs(dataset_path)
    else:
        raise NotADirectoryError("Please input the dataset name correctly")
    
    