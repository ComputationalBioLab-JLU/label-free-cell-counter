import cv2
import os
import pickle
import numpy as np 
import random
from glob import glob
from tqdm import tqdm
from shutil import copy

def split_img(img_path,size,img_index,suffix,save_path):
    img=cv2.imread(img_path)
    iter_number_x=img.shape[0]//size
    iter_number_y=img.shape[1]//size
    for index_1 in range(iter_number_x):
        x_start,x_end = index_1*size , (index_1+1)*size
        for index_2 in range(iter_number_y):
            y_start,y_end = index_2*size , (index_2+1)*size
            window=img[x_start:x_end,y_start:y_end,:]
            cv2.imwrite('{}/{}{}-{}-{}.jpg'.format(save_path,suffix,img_index,index_1,index_2),window)

P_SUFFIX='P_'
OTHER_SUFFIX_LIST=['CB_','CG_','CR_','B_','G_','R_','RGB_']

test_data_path='/state/heyang/ZHEDA_PROJECT_DATA/datas/tcps_data/train_data/test/data/'
small_test_save_path='/state/heyang/ZHEDA_PROJECT_DATA/datas/tcps_data/train_data/test/test_imgs/'
if not os.path.exists(small_test_save_path):
    os.makedirs(small_test_save_path)

p_img_list=glob('{}{}*'.format(test_data_path,P_SUFFIX))
for p_path in tqdm(p_img_list) :
    img_index=os.path.basename(p_path).split('_')[1].split('.')[0]
    split_img(p_path,1144,img_index,P_SUFFIX,small_test_save_path)
    for suffix in OTHER_SUFFIX_LIST:
        path=p_path.replace(P_SUFFIX,suffix)
        print(path)
        split_img(path,1144,img_index,suffix,small_test_save_path)

# sampling_list=[]
# for img_index in ['10W','20W','30W'] :
#     P_list=glob('{}/{}{}*'.format(small_test_save_path,P_SUFFIX,img_index))
#     temp_list=random.sample(P_list,25)
#     sampling_list.extend(temp_list)

# if not os.path.exists(sampling_test_data):
#     os.makedirs(sampling_test_data)

# for sample_path in sampling_list:
#     file_name=os.path.basename(sample_path)
#     copy(sample_path,sampling_test_data+file_name)
#     for suffix in OTHER_SUFFIX_LIST:
#         new_file_name=file_name.replace(P_SUFFIX,suffix)
#         copy(small_test_save_path+new_file_name,sampling_test_data+new_file_name)


