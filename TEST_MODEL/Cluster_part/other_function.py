import os
import cv2
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
def if_zero(x):
    if x < 0:
        return(0)
    else :
        return(x)

def cut_cell_window(centers_dict,cell_img,threshold,show_center,color):
    centers_list=[]
    output_list=[]
    normal_x_size,normal_y_size,_=cell_img.shape
    for key in centers_dict:
        center_x,center_y=centers_dict[key][0]
        center_x,center_y=int(center_x),int(center_y)
        start_x,end_x=if_zero(center_x-threshold),if_zero(center_x+threshold)
        start_y,end_y=if_zero(center_y-threshold),if_zero(center_y+threshold)
        temp_window=cell_img[start_x:end_x,start_y:end_y,:]

        if temp_window.shape[0]!=2*threshold or temp_window.shape[1]!=2*threshold:
            #print(start_x,end_x,start_y,end_y)
            #print(temp_window.shape)
            padding_number_1 = 2*threshold - temp_window.shape[0]
            padding_number_2 = 2*threshold - temp_window.shape[1]
            #print(padding_number_1,padding_number_2)
            padding_list = [0,0,0,0]
            if start_x == 0 :
                padding_list[0]=padding_number_1
            if end_x >= normal_x_size :
                padding_list[1]=padding_number_1
            if start_y == 0 :
                padding_list[2]=padding_number_2
            if end_y >= normal_y_size :
                padding_list[3]=padding_number_2               
            top, bottom, left, right = padding_list
            #print(top, bottom, left, right)
            temp_window = cv2.copyMakeBorder(temp_window,top, bottom, left, right,cv2.BORDER_DEFAULT)
            #print(temp_window.shape)
            #print('$$$$$$$$$$$$$$$$$$$$$')
        if show_center:
            cv2.circle(temp_window,(threshold,threshold),8,color)
        centers_list.append('{}-{}'.format(center_x,center_y))
        output_list.append(temp_window)
    return(centers_list,output_list)

def load_pkl(path):
    input_file=open(path,'rb')
    output=pickle.load(input_file)
    return(output)

def save_windows_list(SAVE_PATH,centers_list,windows_list,save_suffix):
    for index,window in enumerate(windows_list):
        WINDOW_SAVE_PATH='{}/{}{}_{}.jpg'.format(SAVE_PATH,save_suffix,centers_list[index],index)
        cv2.imwrite(WINDOW_SAVE_PATH,window)

def cut_and_save_windows(centers_dict,img,SAVE_PATH,save_suffix,show_center,threshold,color=(100,100,200)):
    centers_list,windows_list=cut_cell_window(centers_dict,img,threshold,show_center,color)
    save_windows_list(SAVE_PATH,centers_list,windows_list,save_suffix)

def split_nuclear_clusters(clusters_dict,low_bound,up_bound):
    small_dict={}
    normal_dict={}
    huge_dict={}
    for key in clusters_dict:
        _,points=clusters_dict[key]
        points_number=len(points)
        if points_number <= low_bound :
            small_dict[key]=clusters_dict[key]
        elif points_number > up_bound :
            huge_dict[key]=clusters_dict[key]
        else:
            normal_dict[key]=clusters_dict[key]
    return(small_dict,normal_dict,huge_dict)

