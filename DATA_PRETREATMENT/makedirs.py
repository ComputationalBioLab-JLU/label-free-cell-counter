import os
import cv2
import random
from shutil import  copy
from glob import glob


F_file_list=['train','val','test']
S_file_list=['data']

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_pic(pic_path):
    test_img_list = []
    train_img_list = []
    img = cv2.imread(pic_path)
    x_size,y_size,_ = img.shape
    test_img = img[:x_size//2,:,:]
    train_img = img[x_size//2:,:,:]
    return(train_img,test_img)
    
BASE_PATH='/state/heyang/ZHEDA_PROJECT_DATA/datas/tcps_data/'
makedirs(BASE_PATH)

for F_filename in F_file_list:
    makedirs('{}/train_data/{}'.format(BASE_PATH,F_filename))
    for S_filename in S_file_list:
        makedirs('{}/train_data/{}/{}'.format(BASE_PATH,F_filename,S_filename))

PIC_SUFFIX='P_'
MASK_SUFFIX_LIST=['P_','CB_','CG_','CR_','B_','G_','R_','RGB_']

pic_list=glob('{}/splited_data/{}*'.format(BASE_PATH,PIC_SUFFIX))
#random.shuffle(pic_list)
#print(pic_list)
for index,PIC_PATH in enumerate(pic_list):
    PIC_INDEX = os.path.basename(PIC_PATH).split('.')[0].split('_')[1]
    for MASK_SUFFIX in MASK_SUFFIX_LIST:
        MASK_PATH=PIC_PATH.replace(PIC_SUFFIX,MASK_SUFFIX)
        train_img,test_img = split_pic(MASK_PATH)
        cv2.imwrite('{}/train_data/train/data/{}{}.jpg'.format(BASE_PATH,MASK_SUFFIX,index),train_img)
        cv2.imwrite('{}/train_data/test/data/{}{}.jpg'.format(BASE_PATH,MASK_SUFFIX,index),test_img)


#        if '-1' in PIC_INDEX or '-2' in PIC_INDEX:
#            copy(MASK_PATH,BASE_PATH+'train_data/train/data/')
#            copy(PIC_PATH,BASE_PATH+'train_data/train/data/')
#        else:
#            copy(MASK_PATH,BASE_PATH+'train_data/test/data/')
#            copy(PIC_PATH,BASE_PATH+'train_data/test/data/')

