import os
import torch
import numpy as np
import pickle
import cv2
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from glob import glob
from tqdm import tqdm

class resnet_model():
    def __init__(self,model_path,cut_size):
        self.cut_size=cut_size
        self.model=self.load_resnet_model(model_path)    
    def make_model(self,pretrained=True):
        model=torchvision.models.resnet18(pretrained=pretrained)
        num_ftrs=model.fc.in_features
        model.fc=nn.Sequential(nn.Linear(num_ftrs,100),
                               nn.Linear(100,1))
        # model.fc=torch.nn.Sequential(nn.Linear(num_ftrs,2))
        for param in model.parameters():
            param.requires_grad=False
        return(model)
    def load_resnet_model(self,path):
        model=self.make_model()
        model=model.cuda()
        model=nn.DataParallel(model)
        model.load_state_dict(torch.load(path),False)
        model.eval()
        return(model)
    def load_resnet_model_(self,path):
        model = torch.load(path)
        model=model.cuda()
        model=nn.DataParallel(model)
        model.eval()
        return(model)
    def transform_pic(self,img_array,cut_size):
        transform=transforms.Compose([transforms.CenterCrop(cut_size),
                                      transforms.ToTensor()])
        pic=Image.fromarray(img_array)
        pic_tensor=transform(pic)
        pic_tensor=pic_tensor.unsqueeze(0)
        return(pic_tensor)
    def draw_circles(self,labels_dict,img_path,save_path):
        source_img=cv2.imread(img_path)
        for key in labels_dict :
            center_x,center_y=os.path.basename(key).split('_')[1].split('-')[-2:]
            if int(labels_dict[key]==0):
                color=(200,0,255)
            else:
                color=(200,255,0)
            cv2.circle(source_img,(int(center_y),int(center_x)),10,color,thickness = -1)
        print(img_path)
        print(save_path+'/pred_'+os.path.basename(img_path))
        cv2.imwrite(save_path+'/pred_'+os.path.basename(img_path),source_img)
    def __call__(self,source_img_path,img_file_path,suffix,pred_save_path):
        self.model.eval()        
        labels_dict={}
        result_dict={}
        for img_path in glob('{}{}*'.format(img_file_path,suffix)):
            img=Image.open(img_path)
            #print(np.asarray(img).shape)
            img_array=np.asarray(img)
            img_tensor=self.transform_pic(img_array,self.cut_size)
            img_tensor=img_tensor.cuda()
            output=self.model(img_tensor)
            output = nn.Sigmoid()(output)
            output = output.detach().cpu().numpy()[0,0]
            if output >= 0.5 :
                pred = 1
            else:
                pred = 0
            labels_dict[img_path]=pred
            if pred not in result_dict:
                result_dict[pred]=1
            else:
                result_dict[pred]+=1
# ======================================================
#            _,pred=torch.max(output,1)
#            labels_dict[img_path]=pred.item()
#            if pred.item() not in result_dict:
#                result_dict[pred.item()]=1
#            else:
#                result_dict[pred.item()]+=1            
# =======================================================  
        self.draw_circles(labels_dict,source_img_path,pred_save_path)  
        return(result_dict)               
    def test_pic(self,img_file_path,suffix):
        result_dict={}
        for img_path in tqdm(glob('{}{}*'.format(img_file_path,suffix))):
            img=Image.open(img_path)
            img_array=np.asarray(img)
            img_tensor=self.transform_pic(img_array,self.cut_size)
            img_tensor=img_tensor.cuda()
            output=self.model(img_tensor)
            output = nn.Sigmoid()(output)
            output = output.detach().cpu().numpy()[0,0]
            if output >= 0.5 :
                pred = 1
            else:
                pred = 0
            if pred not in result_dict:
                result_dict[pred]=1
            else:
                result_dict[pred]+=1  
        for key in result_dict:
            print(key,result_dict[key])      


if __name__ == '__main__' :
    resnet_path = '/state/heyang/cell_recognition_project/train_part/Resnet_part_new/resnet_model/resnet18_12-16.pth'
    resnet_model=resnet_model(resnet_path,112)
    data_path = '/state/heyang/cell_recognition_project/data/train_resnet_data/resnet_input_data_100_0.5/test/Red/'
    resnet_model.test_pic(data_path,'P_')








