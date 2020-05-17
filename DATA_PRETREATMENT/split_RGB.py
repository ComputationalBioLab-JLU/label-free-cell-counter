import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
import pickle


def make_mask(pic_array,color='B'):
    pic_array=np.expand_dims(pic_array,axis=2)
    pic_mask=np.repeat(pic_array,3,axis=2)
    if color=='CB':
        pic_mask[:,:,1:]=pic_mask[:,:,1:]*0
    elif color=='CG':
        pic_mask[:,:,0]=pic_mask[:,:,0]*0
        pic_mask[:,:,2]=pic_mask[:,:,2]*0
    elif color=='CR':
        pic_mask[:,:,:2]=pic_mask[:,:,:2]*0
    else:
        pic_gray=cv2.cvtColor(pic_mask,cv2.COLOR_BGR2GRAY)
        _,pic_mask=cv2.threshold(pic_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        pic_mask=cv2.adaptiveThreshold(pic_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,501,10)
    return(pic_mask)

#==========================================================================
PKL_SUFFIX='.pkl'
B_suffix='B_'
R_suffix='R_'
G_suffix='G_'
PIC_suffix='P_'

BASE_PATH='/state/heyang/ZHEDA_PROJECT_DATA/datas/tcps_data/'
SAVE_PATH='{}/splited_data/'.format(BASE_PATH)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
PKL_PATH='{}/pkl_data/*{}'.format(BASE_PATH,PKL_SUFFIX)
PKL_list=glob.glob(PKL_PATH)
for PKL_path in tqdm(PKL_list):
    PIC_NUMBER=os.path.basename(PKL_path).split('.')[0]
    pkl_file=pickle.load(open(PKL_path,'rb'))
    PIC_array=pkl_file[:,:,:3]
    B_channel=pkl_file[:,:,3]
    G_channel=pkl_file[:,:,4]
    R_channel=pkl_file[:,:,5]
    BGR_channel=pkl_file[:,:,3:]

    cv2.imwrite('{}RGB_{}.jpg'.format(SAVE_PATH,PIC_NUMBER),BGR_channel)
    cv2.imwrite('{}{}{}.jpg'.format(SAVE_PATH,PIC_suffix,PIC_NUMBER),PIC_array)
    cv2.imwrite('{}C{}{}.jpg'.format(SAVE_PATH,B_suffix,PIC_NUMBER),make_mask(B_channel,color='CB')) 
    cv2.imwrite('{}C{}{}.jpg'.format(SAVE_PATH,G_suffix,PIC_NUMBER),make_mask(G_channel,color='CG'))
    cv2.imwrite('{}C{}{}.jpg'.format(SAVE_PATH,R_suffix,PIC_NUMBER),make_mask(R_channel,color='CR'))    
    cv2.imwrite('{}{}{}.jpg'.format(SAVE_PATH,PIC_suffix,PIC_NUMBER),PIC_array)

    cv2.imwrite('{}{}{}.jpg'.format(SAVE_PATH,B_suffix,PIC_NUMBER),make_mask(B_channel,color=None))
    cv2.imwrite('{}{}{}.jpg'.format(SAVE_PATH,G_suffix,PIC_NUMBER),make_mask(G_channel,color=None))
    cv2.imwrite('{}{}{}.jpg'.format(SAVE_PATH,R_suffix,PIC_NUMBER),make_mask(R_channel,color=None))
 



    





 
