import numpy as np
from PIL import Image
import matplotlib.pyplot as PIL
import cv2
from glob import glob
import random
from copy import deepcopy
import torch
import torchvision
from torchvision import transforms

class dataset():
    def __init__(self,data_path,x_suffix,y_suffix,batch_size,shuffle=True):
        self.data_path=data_path
        self.x_suffix=x_suffix
        self.y_suffix=y_suffix
        self.batch_size=int(batch_size)
        self.shuffle=shuffle
        self.id_list=self.get_ids(self.data_path,self.x_suffix)
        self.transform=transforms.Compose([transforms.ToTensor()])
    def __call__(self):
        batch_data=next(self.get_batch())
        return(batch_data)
    def get_ids(self,path,suffix):
        return(glob('{}/{}*'.format(path,suffix)))
    def transform_pic(self,img_path):
        pic=Image.open(img_path)
        pic_tensor=self.transform(pic)
        pic_tensor=pic_tensor.unsqueeze(0)
        return(pic_tensor)
    def load_data(self,data_list):
        init_x_path=data_list[0]
        init_y_path=init_x_path.replace(self.x_suffix,self.y_suffix)
        init_x_tensor=self.transform_pic(init_x_path)
        init_y_tensor=self.transform_pic(init_y_path)
        for x_path in data_list[1:]:
            y_path=x_path.replace(self.x_suffix,self.y_suffix)
            x_tensor=self.transform_pic(x_path)
            y_tensor=self.transform_pic(y_path)
            init_x_tensor=torch.cat((init_x_tensor,x_tensor),0)
            init_y_tensor=torch.cat((init_y_tensor,y_tensor),0)
        return(init_x_tensor,init_y_tensor)
    def get_batch(self):
        input_data=self.id_list
        batch_size=self.batch_size
        shuffle=self.shuffle
        rows=len(input_data)
        index_list=list(range(rows))
        if shuffle:
            random.shuffle(input_data)
        while True:
            batch_index=index_list[0:batch_size]
            index_list=index_list[batch_size:]+index_list[:batch_size]
            batch_data=[]
            for index in batch_index:
                batch_data.append(input_data[index])
            yield(self.load_data(batch_data))
    def show_data_number(self):
        return len(self.id_list)



