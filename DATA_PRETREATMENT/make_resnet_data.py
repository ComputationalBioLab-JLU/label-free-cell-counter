#=====================================================================
# MAKE_RESNET_DATA.PY SHOULD BE CALL AFTER  
# ../COUNT_LABEL/Cluster_part/cluster_model.py
# CLUSTER_MODEL.PY WILL LOCATED EVERY CELL
#=====================================================================
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import random
import shutil
def center_cut(pic,threshold):
    pic_size=pic.shape[0]
    start,end=pic_size//2-threshold,pic_size//2+threshold

    if len(pic.shape)>2:
        return(pic[start:end,start:end,:])
    else:
        return(pic[start:end,start:end])
def compute_label(R_img,G_img,B_img,threshold=100):
    R_img=center_cut(R_img,threshold)
    G_img=center_cut(G_img,threshold)
    B_img=center_cut(B_img,threshold)
    RB_array=np.multiply(R_img,B_img)
    GB_array=np.multiply(G_img,B_img)
    RB_sum=np.sum(RB_array)
    GB_sum=np.sum(GB_array)
    if RB_sum/(RB_sum+GB_sum)>0.5:
        label='Red'
    elif GB_sum/(GB_sum+RB_sum)>0.5:
        label='Green'
    else:
        label=None
    return(label)
    
def open_mask_img(mask_path):
    mask_img=cv2.imread(mask_path)
    img_array=cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
    img_array=cv2.threshold(img_array,127,255,cv2.THRESH_BINARY)[1]/255
    return(img_array)

P_SUFFIX='P_'
R_SUFFIX='R_'
G_SUFFIX='G_'
B_SUFFIX='B_'
BASE_PATH='/state/heyang/ZHEDA_PROJECT_DATA/datas/tcps_data/train_data/'
for phase in ['train','test']:
    print(phase)
    p=Pool(40)
    p_path_list=glob('{}{}/cluster_windows/*/{}*'.format(BASE_PATH,phase,P_SUFFIX))
    # if phase == 'val':
    #     p_path_list = random.sample(p_path_list,3000)
    for P_PATH in tqdm(p_path_list):
        basename=os.path.basename(P_PATH)
        G_PATH=P_PATH.replace(P_SUFFIX,G_SUFFIX)
        g_img=open_mask_img(G_PATH)
        R_PATH=P_PATH.replace(P_SUFFIX,R_SUFFIX)
        r_img=open_mask_img(R_PATH)
        B_PATH=P_PATH.replace(P_SUFFIX,B_SUFFIX)

        b_img=open_mask_img(B_PATH)
        label=p.apply_async(compute_label,args=(r_img,g_img,b_img,)).get()
        #label=compute_label(r_img,g_img,b_img)
        if label == None :
            continue
        DATA_SAVE_PATH='{}/resnet_input_data/{}/{}'.format(BASE_PATH,phase,label)
        if not os.path.exists(DATA_SAVE_PATH):
            os.makedirs(DATA_SAVE_PATH)
        p_img=cv2.imread(P_PATH)
        cv2.imwrite('{}/{}'.format(DATA_SAVE_PATH,basename.replace('tif','jpg')),p_img)
    p.close()
    p.join()


green_list = []
red_list  = []
train_save_path = '{}/resnet_input_data/train/'.format(BASE_PATH)
val_save_path = '{}/resnet_input_data/val/'.format(BASE_PATH)
for label_path in glob(train_save_path+'*'):
    label = label_path.split('/')[-1]
    temp_list = []
    for data_path in glob('{}{}/P_*'.format(train_save_path,label)):
        temp_list.append(data_path)
    green_list = random.sample(temp_list,500)    

    save_path = '{}{}/'.format(val_save_path,label)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for path in green_list:
        pic_name = os.path.basename(path)
        shutil.move(path,save_path+pic_name)
print('FINISH')



