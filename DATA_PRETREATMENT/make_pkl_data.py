import cv2
import os
import glob
import numpy as np
import pickle
from tqdm import tqdm

def cut_pic(pic_array,ideal_size=572*26):
    output_array=pic_array[:ideal_size,:ideal_size,:]
    return(output_array)


def make_pkl_data(B_PATH,F_PATH,SAVE_PATH):
    IMG_NUMBER=os.path.basename(B_PATH).split(' ')[1]

    B_img=cv2.imread(B_PATH)
    F_img=cv2.imread(F_PATH)

    B_img=cut_pic(B_img)
    F_img=cut_pic(F_img)

    output_img=np.concatenate([B_img,F_img],axis=2)
    OUTPUT_PATH='{}{}.pkl'.format(SAVE_PATH,IMG_NUMBER)
    output_file=open(OUTPUT_PATH,'wb')
    pickle.dump(output_img,output_file)
    output_file.close()



PIC_SUFFIX='W BF_RGB.tif'
MASK_SUFFIX='W_RGB.tif'

BASE_PATH='/state/heyang/ZHEDA_PROJECT_DATA/'
SOURCE_PATH='{}/source_data/PDMS_source_data/'.format(BASE_PATH)
SAVE_PATH='{}/datas/pdms_data/pkl_data/'.format(BASE_PATH)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
pic_list=glob.glob('{}*{}'.format(SOURCE_PATH,PIC_SUFFIX))
for B_PATH in (pic_list):
    print(B_PATH)
    F_PATH=B_PATH.replace(PIC_SUFFIX,MASK_SUFFIX)
    make_pkl_data(B_PATH,F_PATH,SAVE_PATH)


