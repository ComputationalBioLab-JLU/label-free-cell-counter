
import torch
import torch.nn.functional as fun
import torch.nn as nn
#import Unet_part as parts
from .Unet_part import *

class unet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(unet,self).__init__()
        self.input_layer=double_conv(n_channels,64)
        self.down_1=down_sampling(64,128)
        self.down_2=down_sampling(128,256)
        self.down_3=down_sampling(256,512)
        self.down_4=down_sampling(512,1024)
        self.up_4=up_sampling(1024,512)
        self.up_3=up_sampling(512,256)
        self.up_2=up_sampling(256,128)
        self.up_1=up_sampling(128,64)
        self.output_layer=output_layer(64,n_classes)
    def forward(self,x):
        x1=self.input_layer(x)
#        print('x1 : ',x1.shape)
        x2=self.down_1(x1)
#        print('x2 : ',x2.shape)
        x3=self.down_2(x2)
#        print('x3 : ',x3.shape)
        x4=self.down_3(x3)
#        print('x4 : ',x4.shape)
        x5=self.down_4(x4)
#        print('x5 : ',x5.shape)
        x=self.up_4(x5,x4)
#        print('u4 : ',x.shape)
        x=self.up_3(x,x3)
#        print('u3 : ',x.shape)
        x=self.up_2(x,x2)
#        print('u2 : ',x.shape)
        x=self.up_1(x,x1)
#        print('u1 : ',x.shape)
        x=self.output_layer(x)
#        print('out : ',x.shape)
        return(torch.sigmoid(x))

