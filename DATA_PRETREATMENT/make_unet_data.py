import os
import sys
import cv2
import shutil
from random import sample
from glob import glob
from tqdm import tqdm

DATA_SUFFIX='P_'
MASK_SUFFIX='B_'
SAVE_DATA_SUFFIX='PIC_'
SAVE_MASK_SUFFIX='MASK_'
SPLIT_SIZE=572
STRIDE=572

def split_data(pic_path,save_path,split_size,stride,save_suffix):
    img=cv2.imread(pic_path)
    x_size,y_size,channel=img.shape
    x_index_number=(x_size-split_size)//stride+1
    y_index_number=(y_size-split_size)//stride+1
    for x_index in range(x_index_number):
        x_start,x_end=x_index*stride,x_index*stride+split_size
        for y_index in range(y_index_number):
            y_start,y_end=y_index*stride,y_index*stride+split_size
            window_img=img[x_start:x_end,y_start:y_end,:]
            cv2.imwrite('{}/{}{}-{}.jpg'.format(save_path,save_suffix,x_index,y_index),window_img)

BASE_PATH='/state/heyang/ZHEDA_PROJECT_DATA/datas/tcps_data/train_data/'
for phase in ['train','val','test']:
    print(phase)
    data_list=glob('{}{}/data/{}*'.format(BASE_PATH,phase,DATA_SUFFIX))
    save_path='{}{}/unet_windows/'.format(BASE_PATH,phase)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for img_path in tqdm(data_list):
        img_index=os.path.basename(img_path).split('.')[0].split('_')[1]
        split_data(img_path,save_path, split_size=SPLIT_SIZE,stride=STRIDE,save_suffix=SAVE_DATA_SUFFIX+img_index+'_')
        mask_path=img_path.replace(DATA_SUFFIX,MASK_SUFFIX)
        split_data(mask_path, save_path, split_size=SPLIT_SIZE,stride=STRIDE,save_suffix=SAVE_MASK_SUFFIX + img_index + '_')

data_list = []
for data_path in glob('{}/train/unet_windows/{}*'.format(BASE_PATH,SAVE_DATA_SUFFIX)):
    data_list.append(data_path)
val_data_list = sample(data_list,100)
for data_path in val_data_list:
    shutil.move(data_path,'{}/val/unet_windows/'.format(BASE_PATH))
    shutil.move(data_path.replace(SAVE_DATA_SUFFIX,SAVE_MASK_SUFFIX),'{}/val/unet_windows/'.format(BASE_PATH))

