

import torch
import torch.nn as nn
import torch.nn.functional as fun

class double_conv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(double_conv,self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_channel,out_channel,3,padding=1),
                                nn.BatchNorm2d(out_channel),
                                nn.ReLU(inplace=True),                               #inplace=True 输出结果直接覆盖输入，不需要额外保存
                                nn.Conv2d(out_channel,out_channel,3,padding=1),
                                nn.BatchNorm2d(out_channel),
                                nn.ReLU(inplace=True))
    def forward(self,x):
        return(self.conv(x))
class down_sampling(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(down_sampling,self).__init__()
        self.conv=nn.Sequential(nn.MaxPool2d(2),
                                double_conv(in_channel,out_channel))
    def forward(self,x):
        return(self.conv(x))
class up_sampling(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(up_sampling,self).__init__()
        self.conv_1=nn.ConvTranspose2d(in_channel,in_channel//2,2,stride=2)
        self.conv_2=double_conv(in_channel,out_channel)
    def forward(self,x1,x2):
        x1=self.conv_1(x1)
        diff_x=x2.shape[2]-x1.shape[2]                          #input size is Batch,channel,height,weight
        diff_y=x2.shape[3]-x1.shape[3]
        x1=fun.pad(x1,(diff_x//2,diff_x-diff_x//2,diff_y//2,diff_y-diff_y//2))    #将上采样得到的数据集扩大而不是裁剪下采样的数据？
        
        x=torch.cat([x2,x1],dim=1)                              #合并数据集
        x=self.conv_2(x)
        return(x)
class output_layer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(output_layer,self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_channel,out_channel,1))
        #self.conv=nn.Conv2d(in_channel,out_channel,1)
        
    def forward(self,x):
        return(self.conv(x))        




