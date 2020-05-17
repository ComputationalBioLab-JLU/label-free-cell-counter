import sys
sys.path.append('../')
import os
import numpy as np
import random
import cv2
import pickle
from glob import glob
from tqdm import tqdm
from Cluster_part.DBSCAN_cluster import *
from Cluster_part.other_function import *

class cluster_model():
    def __init__(self,eps,min_samples,bandwidth,circle_color,circle_size,low_bound,up_bound):
        self.circle_color=circle_color
        self.circle_size=circle_size
        self.low_bound=low_bound
        self.up_bound=up_bound 
        self.DB_model=DBSCAN_model(eps,min_samples)
        self.MS_model=Meanshift_model(bandwidth)
    def draw_circles(self,center_dict,img,color,circle_size):
        for key in center_dict:
             center_x,center_y=center_dict[key][0]
             cv2.circle(img,(int(center_y),int(center_x)),circle_size,color,thickness = -1)
        return(img)
    def __call__(self,mask_path,pic_path,mask_suffix,pic_suffix,window_save_path,result_save_path,cut_size,draw_list):
        if not os.path.exists(result_save_path) :
            os.makedirs(result_save_path)
        PIC_INDEX=os.path.basename(mask_path).split('.')[0].split('_')[1]
        temp_windows_save_path='{}{}/'.format(window_save_path,PIC_INDEX)
        if not os.path.exists(temp_windows_save_path):
            os.makedirs(temp_windows_save_path)
        mask_img=cv2.imread(mask_path)
        noise_dict,normal_dict,mult_dict=split_nuclear_clusters(self.DB_model(mask_img),low_bound=self.low_bound,up_bound=self.up_bound)
        split_mult_dict={}
        normal_dict.update(mult_dict)                                                                  

        for suffix in tqdm(draw_list):
            img=cv2.imread(pic_path.replace(pic_suffix,suffix))
            print(pic_path.replace(pic_suffix,suffix))
            cir_img=self.draw_circles(normal_dict,img,color=self.circle_color,circle_size=self.circle_size)
            cir_img=self.draw_circles(split_mult_dict,cir_img,color=(0,255,0),circle_size=self.circle_size)
            cv2.imwrite('{}/point_{}{}.jpg'.format(result_save_path,suffix,PIC_INDEX),cir_img)
            cut_and_save_windows(normal_dict,img,temp_windows_save_path,save_suffix=suffix,show_center=False,threshold=cut_size)
            
        p_img=cv2.imread(pic_path)
        cut_and_save_windows(normal_dict,p_img,temp_windows_save_path,save_suffix=pic_suffix,show_center=True,threshold=cut_size)



if __name__=='__main__':

    PIC_PATH='/state/heyang/cell_recognition_project/reinit_learning_part/PDMS_DATA/transfor_data/'
    P_SUFFIX='P_'
    RGB_SUFFIX='RGB_'
    B_SUFFIX='B_'
    G_SUFFIX='G_'
    R_SUFFIX='R_'
    model=cluster_model(eps=10,
                        min_samples=200,
                        bandwidth=23,
                        circle_color=(0,0,255),
                        circle_size=5,
                        low_bound=500,
                        up_bound=1500)
    for phase in ['train','test']:
        print(phase)
        mask_path_list = glob('{}/{}/data/{}*'.format(PIC_PATH,phase,B_SUFFIX))
        for mask_path in tqdm(mask_path_list):
            pic_path = mask_path.replace(B_SUFFIX,P_SUFFIX)
            window_save_path = '{}/{}/cluster_windows/'.format(PIC_PATH,phase)
            result_save_path = '{}/{}/cluster_result/'.format(PIC_PATH,phase)
            draw_list = [P_SUFFIX,B_SUFFIX,G_SUFFIX,R_SUFFIX,RGB_SUFFIX]
            model(mask_path,pic_path,B_SUFFIX,P_SUFFIX,window_save_path,result_save_path,112,draw_list)
