import os
import sys
import configparser
import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm
from unet_nuclear_pred import Unet_model as unet_model
from Cluster_part.cluster_model import cluster_model
from load_Resnet import resnet_model
class pred_model():
    def __init__(self,config_path):
        conf=configparser.ConfigParser()
        conf.read(config_path)
        self.DATA_PATH=conf.get('input_data','Data_path')           
        self.UNET_MODEL_PATH=conf.get('model_path','Unet_model_path') 
        self.RESNET_MODEL_PATH=conf.get('model_path','Resnet_model_path')
        self.P_SUFFIX=conf.get('data_suffix','Pic_suffix')
        self.RGB_SUFFIX=conf.get('data_suffix','RGB_suffix')
        self.UNET_SUFFIX=conf.get('data_suffix','Unet_pred_suffix')
        self.BASE_SAVE_PATH=conf.items('output_data')[0][1]
        self.filename_list=conf.items('output_data')[1:]
        self.UNET_PRED_PATH,self.CLUSTER_PRED_PATH,self.CLUSTER_WINDOWS_PATH,self.RESNET_PRED_PATH=self.init_dirs(self.BASE_SAVE_PATH,self.filename_list)
        self.Unet_model=unet_model(self.UNET_MODEL_PATH)
        DBSCAN_EPS=float(conf.get('Cluster_parameter','DBSCAN_eps'))
        DBSCAN_MIN_SAMPLES=float(conf.get('Cluster_parameter','DBSCAN_min_samples'))
        MEANSHIFT_BANDWIDTH=float(conf.get('Cluster_parameter','Meanshift_bandwidth'))
        LOW_BOUND=int(conf.get('Cluster_parameter','Low_bound'))
        UP_BOUND=int(conf.get('Cluster_parameter','Up_bound'))
        CIRCLE_COLOR=(0,0,255)
        CIRCLE_SIZE=10
        self.CUT_SIZE=int(conf.get('Cluster_parameter','cut_size'))
        self.RESNET_CUT_SIZE=int(conf.get('resnet_parameter','cut_size'))
        B_suffix=conf.get('data_suffix','B_suffix')
        CB_suffix=conf.get('data_suffix','CB_suffix')
        self.DRAW_LIST=[B_suffix,CB_suffix,self.P_SUFFIX]
        self.cluster_model=cluster_model(DBSCAN_EPS,DBSCAN_MIN_SAMPLES,MEANSHIFT_BANDWIDTH,CIRCLE_COLOR,CIRCLE_SIZE,LOW_BOUND,UP_BOUND)
        self.resnet_model=resnet_model(self.RESNET_MODEL_PATH,self.RESNET_CUT_SIZE)
    def init_dirs(self,base_path,filename_list):
        path_list=[]
        self.make_dirs(base_path)
        for filename in filename_list:
            temp_path='{}/{}/'.format(base_path,filename[1])
            path_list.append(self.make_dirs(temp_path))
        return(path_list)
    def make_dirs(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        return(path)    
    def Cluster_step(self,UNET_SAVE_PATH,PIC_PATH):
        self.cluster_model(UNET_SAVE_PATH,
                           PIC_PATH,
                           self.UNET_SUFFIX,
                           self.P_SUFFIX,
                           self.CLUSTER_WINDOWS_PATH,
                           self.CLUSTER_PRED_PATH,
                           self.CUT_SIZE,
                           self.DRAW_LIST)
    def __call__(self):
        print('{}{}*'.format(self.DATA_PATH,self.P_SUFFIX))
        result_dict = {}
        for PIC_PATH in tqdm(glob('{}{}*'.format(self.DATA_PATH,self.P_SUFFIX))):
            result_array = np.zeros(2)
            PIC_INDEX=os.path.basename(PIC_PATH).split('_')[1].split('.')[0]
            UNET_SAVE_PATH=PIC_PATH.replace(self.DATA_PATH,self.UNET_PRED_PATH).replace(self.P_SUFFIX,self.UNET_SUFFIX)
            Unet_pred=self.Unet_model(PIC_PATH, UNET_SAVE_PATH)
            print(PIC_PATH)
            self.Cluster_step(UNET_SAVE_PATH,PIC_PATH)
            WINDOWS_PATH = '{}{}/'.format(self.CLUSTER_WINDOWS_PATH,PIC_INDEX)
            RGB_PATH=PIC_PATH.replace(self.P_SUFFIX,self.RGB_SUFFIX)
            temp_result_dict=self.resnet_model(RGB_PATH,WINDOWS_PATH,self.P_SUFFIX,self.RESNET_PRED_PATH)
            for key in temp_result_dict:
                result_array[key] = temp_result_dict[key]
            result_dict[PIC_INDEX] = result_array.tolist()
        save_path = '{}/test_pred.js'.format(self.BASE_SAVE_PATH)
        save_file = open(save_path,'w')
        json.dump(result_dict,save_file)
        save_file.close()
a=pred_model('./pred_config')
a()
