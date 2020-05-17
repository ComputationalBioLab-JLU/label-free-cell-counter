# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:00:38 2020

@author: 10979
"""

import cv2
import os
from glob import glob 
import numpy as np
import random
import shutil
import xlrd

def split_data(img_path,size = 1144):
    img_name = os.path.basename(img_path).split('.')[0]
    suffix = img_name.split('_')[0]
    img_index = img_name.split('_')[1]
    save_path = '/state/heyang/ZHEDA_PROJECT_DATA/datas/old_data/train_data/test/small_count_data/{}'.format(img_index)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = cv2.imread(img_path)
    shape_x,shape_y,_ = np.shape(img)
    all_index_x = shape_x//size
    all_index_y = shape_y//size
    print(shape_x,shape_y)
    for index_x in range(all_index_x):
        x_start,x_end = index_x*size,(index_x+1)*size
        for index_y in range(all_index_y):
            y_start,y_end = index_y*size,(index_y+1)*size
            small_img = img[x_start:x_end,y_start:y_end,:]
            cv2.imwrite('{}/{}_{}-{}.jpg'.format(save_path,suffix,index_x,index_y), small_img)

def sampling_data(all_data_path,save_path,number = 25):
    file_path_list = glob('{}/*'.format(all_data_path))
    for file_path in file_path_list:
        file_name = os.path.basename(file_path)
        temp_save_path = '{}/{}/'.format(save_path,file_name)
        os.makedirs(temp_save_path)
        unsample_list = glob('{}/B_*.jpg'.format(file_path))
        sample_list = random.sample(unsample_list,number)
        for img_path in sample_list:
            img_save_path = '{}/{}'.format(temp_save_path,os.path.basename(img_path))
            shutil.copyfile(img_path, img_save_path)

def copy_img(img_path,save_path,suffix_list = ['CB','CR','CG','B','G','R','P','RGB']):
    for suffix in suffix_list:
        temp_img_path = img_path.replace('*',suffix)
        temp_name = os.path.basename(temp_img_path)
        temp_save_path = '{}/{}'.format(save_path,temp_name)
        shutil.copyfile(temp_img_path, temp_save_path)


def read_xlsx_file(file_path,image_path,save_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheets()[0] 
    nrows = table.nrows
    for i in range(1,nrows):
        img_index = table.cell_value(i,0)
        window_index = table.cell_value(i,1)
        img_path = '{}/{}/*_{}.jpg'.format(image_path,img_index,window_index) 
        temp_save_path = '{}/{}/'.format(save_path,img_index)
        if not os.path.exists(temp_save_path):
            os.makedirs(temp_save_path)
        copy_img(img_path, temp_save_path)  

read_xlsx_file(file_path = './count_label_excel.xls',
               image_path = '/state/heyang/ZHEDA_PROJECT_DATA/datas/old_data/train_data/test/small_count_data/',
               save_path = '/state/heyang/ZHEDA_PROJECT_DATA/datas/old_data/train_data/test/sampling_datas/')


    
#img_list = glob('/state/heyang/ZHEDA_PROJECT_DATA/datas/old_data/train_data/test/data/*.jpg')
#
#for img_path in img_list:    
#    split_data(img_path)




