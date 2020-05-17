import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import pickle
from sklearn.cluster import DBSCAN
from glob import glob
from tqdm import tqdm

class DBSCAN_model():
    def __init__(self,eps,min_samples):
        self.model=DBSCAN(eps,min_samples)
    def run_cluster(self,points_data):
        cluster_result=self.model.fit(points_data)
        return(cluster_result.labels_)
    def fix_cluster_result(self,xy_array,labels):
        cluster_dict={}
        center_dict={}
        xy_list=list(xy_array)
        for index,value in enumerate(xy_list):
            label=labels[index]
            if label==-1:continue
            value=list(value)
            if label not in cluster_dict:
                cluster_dict[label]=[value]
            else:
                cluster_dict[label].append(value)
        for key in cluster_dict:
            cluster_dict[key]=[self.compute_center(cluster_dict[key]),cluster_dict[key]]
        return(cluster_dict)
    def compute_center(self,point_list):
        total_x=0
        total_y=0
        num=len(point_list)
        for point in point_list:
            total_x+=point[0]
            total_y+=point[1]
        return([total_x//num,total_y//num])    
    def make_points_data(self,mask_array):
        data_list=[]
        x_shape,y_shape=mask_array.shape
        for x_index in (range(x_shape)):
            for y_index in range(y_shape):
                if mask_array[x_index,y_index]==1:
                    data_list.append([x_index,y_index])
        data_array=np.array(data_list)
        return(data_array)
    def __call__(self,img_array):
        img_array=cv2.threshold(img_array,100,255,cv2.THRESH_BINARY)[1]
        img_array=img_array[:,:,0]/255
        xy_array=self.make_points_data(img_array)
        cluster_list=self.run_cluster(xy_array)
        cluster_dict=self.fix_cluster_result(xy_array,cluster_list)
        return(cluster_dict)


