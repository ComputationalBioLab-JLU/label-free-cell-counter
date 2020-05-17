import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import math
import pickle
from glob import glob
from tqdm import tqdm
from Unet_model import unet
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import cv2


class Unet_model():
    def __init__(self,model_path):
        self.unet_model=self.load_unet_model(model_path)
    def __call__(self,img_path,save_path):
        img=cv2.imread(img_path)
        img_array=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        pic_dict=self.slice_pic(img_array,572,572,self.unet_model)
        pred_array=self.joint_pic(pic_dict)
        pred_array=(pred_array/255)>0.5
        pred_array=np.uint8(pred_array*255)
        pred_img=Image.fromarray(pred_array)
        pred_img.save(save_path)
    def load_unet_model(self,model_path):
        if 'pkl' in model_path:
            unet_model=torch.load(model_path)
            unet_model=unet_model.cuda()
            unet_model=nn.DataParallel(unet_model)
            return(unet_model)
        else:
            unet_model=unet(3,3)
            unet_model=unet_model.cuda()
            unet_model=nn.DataParallel(unet_model)
            unet_model.load_state_dict(torch.load(model_path))
            return(unet_model)
    def transform_pic(self,img_array):
        transform=transforms.Compose([transforms.ToTensor()])
        pic=Image.fromarray(img_array)
        pic_tensor=transform(pic)
        pic_tensor=pic_tensor.unsqueeze(0)
        return(pic_tensor)
    def transform_tensor(self,tensor):
        tensor=tensor.cpu()[0]
        pil_img=transforms.ToPILImage()(tensor.float()).convert('RGB')
        return(pil_img)
    def slice_pic(self,pic_array,slice_size,stride,unet_model):
        output_dict={}
        x_size,y_size,channel=pic_array.shape
        x_side_number=(x_size-slice_size)//stride+1
        y_side_number=(y_size-slice_size)//stride+1
        for x_index in (range(x_side_number)):
            x_start,x_end=x_index*stride,x_index*stride+slice_size
            for y_index in range(y_side_number):
                y_start,y_end=y_index*stride,y_index*stride+slice_size
                temp_window=pic_array[x_start:x_end,y_start:y_end,:]
                img_tensor=self.transform_pic(temp_window)
                img_tensor=img_tensor.cuda()
                output_tensor=self.unet_model(img_tensor)
                output_array=np.asarray(self.transform_tensor(output_tensor))
                output_dict['{}-{}'.format(x_index,y_index)]=output_array
        return(output_dict)
    def joint_pic(self,pic_dict):
        side_index_number=int(math.sqrt(len(pic_dict)))
        pic_lines=[]
        for x_index in range(side_index_number):
            init_x_array=pic_dict['{}-0'.format(x_index)]
            for y_index in range(1,side_index_number):
                temp_window=pic_dict['{}-{}'.format(x_index,y_index)]    
                init_x_array=np.concatenate((init_x_array,temp_window),axis=1)
            pic_lines.append(init_x_array)
        init_y_line=pic_lines[0]
        for line in pic_lines[1:]:
            init_y_line=np.concatenate((init_y_line,line),axis=0)
        return(init_y_line)

#def load_pkl_data(pkl_piath):
#    input_file=open(pkl_path,'rb')
#    img_array=pickle.load(input_file)
#    return(img_array)


def test_output(data_path,data_suffix,copy_suffix_list,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model=Unet_model('/state/heyang/cell_recognition_project/train_part/Unet_part/unet_model/Unet_iter_74000.pth')
    data_list=glob('{}/{}*'.format(data_path,data_suffix))
    for data_path in tqdm(data_list):
        for suffix in copy_suffix_list:
            img=cv2.imread(data_path.replace(data_suffix,suffix))
            cv2.imwrite('{}/{}'.format(save_path,os.path.basename(data_path).replace(data_suffix,suffix)),img)
        save_output_path='{}/Unet_{}'.format(save_path,os.path.basename(data_path)[3:])
        model(data_path,save_output_path) 
     
DATA_PATH='/state/heyang/cell_recognition_project/transfer_learning_part/PDMS_DATA/transfor_data_noms/train/unet_windows/'
PRED_SAVE_PATH='/state/heyang/cell_recognition_project/transfer_learning_part/PDMS_DATA/transfor_data_noms/train/unet_windows/'

if not os.path.exists(PRED_SAVE_PATH):
    os.makedirs(PRED_SAVE_PATH)
test_output(DATA_PATH,'PIC_',[],PRED_SAVE_PATH)


