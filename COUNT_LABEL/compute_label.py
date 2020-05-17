import os
import numpy as np
import cv2
import json
from Cluster_part.cluster_model import cluster_model
from glob import glob
from tqdm import tqdm

def open_mask_img(mask_path):
    mask_img=cv2.imread(mask_path)
    img_array=cv2.cvtColor(mask_img,cv2.COLOR_BGR2GRAY)
    img_array=cv2.threshold(img_array,127,255,cv2.THRESH_BINARY)[1]/255
    return(img_array)
def center_cut(pic,threshold):
    pic_size=pic.shape[0]
    start,end=pic_size//2-threshold,pic_size//2+threshold

    if len(pic.shape)>2:
        return(pic[start:end,start:end,:])
    else:
        return(pic[start:end,start:end])

def compute_label(R_img,G_img,B_img,threshold):
    R_img=center_cut(R_img,threshold)
    G_img=center_cut(G_img,threshold)
    B_img=center_cut(B_img,threshold)
    RB_array=np.multiply(R_img,B_img)
    GB_array=np.multiply(G_img,B_img)
    RB_sum=np.sum(RB_array)
    GB_sum=np.sum(GB_array)
    # print(RB_array.shape,GB_array.shape)
    # print(RB_sum,GB_sum,np.sum(R_img),np.sum(G_img),np.sum(B_img))
    if RB_sum/(RB_sum+GB_sum)>0.5:
        label='Red'
    elif GB_sum/(GB_sum+RB_sum)>0.5:
        label='Green'
    else:
        label=None
    return(label)

#PIC_PATH='/state/heyang/zheda_project_history/zheda_cell_project/Old_data_retrain/data/train_unet_data/test/sampling_test_data/'
PIC_PATH = '/state/heyang/ZHEDA_PROJECT_DATA/datas/old_data/train_data/test/sampling_datas/5-3/'
RGB_SUFFIX='RGB_'
B_SUFFIX='B_'
G_SUFFIX='G_'
R_SUFFIX='R_'
print(PIC_PATH)
RESULT_SAVE_PATH = '/state/heyang/ZHEDA_PROJECT_DATA/datas/old_data/train_data/test/sampling_datas/5-3_count_label/'
model=cluster_model(eps=10,
                    min_samples=200,
                    bandwidth=23,
                    circle_color=(0,0,255),
                    circle_size=5,
                    low_bound=500,
                    up_bound=1200)
B_list=glob('{}{}*'.format(PIC_PATH,B_SUFFIX))
label_dict={}
for B_PATH in tqdm(B_list):
    RGB_PATH=B_PATH.replace(B_SUFFIX,RGB_SUFFIX)
    model(mask_path=B_PATH,
          pic_path=RGB_PATH,
          mask_suffix=B_SUFFIX,
          pic_suffix=RGB_SUFFIX,
          #window_save_path='../count_sample_result/label_result/label_windows/',
          #result_save_path='../count_sample_result/label_result/cluster_result/',

          window_save_path = '{}/count_label_windows/'.format(RESULT_SAVE_PATH),
          result_save_path = '{}/count_cluster_result/'.format(RESULT_SAVE_PATH),
          cut_size=224,
          draw_list=[B_SUFFIX,G_SUFFIX,R_SUFFIX])
    PIC_INDEX=os.path.basename(B_PATH).split('.')[0].split('_')[1]
    WINDOWS_PATH='{}/count_label_windows/{}/'.format(RESULT_SAVE_PATH,PIC_INDEX)
    result_array = np.zeros(2)
    for B_window_path in tqdm(glob('{}{}*'.format(WINDOWS_PATH,B_SUFFIX))):
        B_img=open_mask_img(B_window_path)
        G_img=open_mask_img(B_window_path.replace(B_SUFFIX,G_SUFFIX))
        R_img=open_mask_img(B_window_path.replace(B_SUFFIX,R_SUFFIX))
        label=compute_label(R_img,G_img,B_img,threshold=100)
        if label == 'Green':
            result_array[0]+=1
        else:
            result_array[1]+=1
    label_dict[PIC_INDEX] = result_array.tolist()

label_save_file = open('{}/count_sample_label.js'.format(RESULT_SAVE_PATH),'w')
json.dump(label_dict,label_save_file)
label_save_file.close()
