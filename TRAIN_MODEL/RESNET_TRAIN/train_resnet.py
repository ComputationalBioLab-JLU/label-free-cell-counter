import os
os.system('export TORCH_HOME=$TORCH_HOME:{}'.format('./torch'))
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from torchvision import datasets, models, transforms
from torchsummary import summary


class resnet_model():
    def __init__(self,data_path,batch_size,save_path):
        self.data_path=data_path
        self.model_save_path=save_path
        self.batch_size=batch_size
        self.data_transforms={'train' : transforms.Compose([
                                        transforms.CenterCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor()]),
                              'val' : transforms.Compose([
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])}
        self.image_dataset = {x:datasets.ImageFolder(os.path.join(self.data_path,x),self.data_transforms[x]) for x in ['train','val']}
        self.data_loaders={x:torch.utils.data.DataLoader(self.image_dataset[x],self.batch_size,shuffle=True,) for x in ['train','val']}
        self.dataset_size={x:len(self.image_dataset[x]) for x in ['train','val']}
        self.class_names=self.image_dataset['train'].classes
        self.device=torch.device('cuda:0,1,2,3' if torch.cuda.is_available() else 'cpu')
        self.model=self.load_resnet_model('/state/heyang/OLD_CODE/cell_recognition_project/train_part/Resnet_part_new/resnet_model/resnet18_12-16.pth')
        self.criterion=self.make_criterion()
        self.optimizer=self.make_optimizer(self.model.parameters())
        self.lr_scheduler=self.make_lr_scheduler(self.optimizer)

    def make_model(self,pretrained=True):
        model=torchvision.models.resnet18(pretrained=pretrained)
        num_ftrs=model.fc.in_features
        model.fc=nn.Sequential(nn.Linear(num_ftrs,100),
                               nn.Linear(100,1))
        for param in model.parameters():
            param.requires_grad=True
        return(model)
    def load_resnet_model(self,path):
        model=self.make_model()
        model=model.cuda()
        model=nn.DataParallel(model)
        model.load_state_dict(torch.load(path),False)
        return(model)

    def make_criterion(self):
        return(nn.BCEWithLogitsLoss(reduction='sum'))
    def make_optimizer(self,model_params):
        return(optim.SGD(model_params,lr=0.0005,momentum=0.9))
    def make_lr_scheduler(self,optimizer):
        xp_lr_scheduler=lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=0.0001)
        return(xp_lr_scheduler) 
    def train_model(self,num_epoch):
        sigmoid=nn.Sigmoid()
        model=self.model
        criterion=self.criterion
        optimizer=self.optimizer
        scheduler=self.lr_scheduler
        start_time=time.time()
        best_model=None
        best_acc=0.0
        log_file=open('./reinit_PDMS_train_log_alpha_0.2','w')
        for epoch in range(num_epoch):
            log_file.write('{}   '.format(epoch))
            print('='*50)
            print('Epoch {}/{} start.'.format(epoch,num_epoch))
            for phase in ['train','val']:
                if phase=='train':
                    scheduler.step()
                    model.train()
                    print('Learning rate : ',optimizer.state_dict()['param_groups'][0]['lr'])
                else:
                    model.eval()
                running_loss=0.0
                running_corrects=0
                for inputs,labels in self.data_loaders[phase]:
                    labels = labels.view(-1,1).float()
                    inputs=inputs.to(self.device)
                    labels=labels.to(self.device)
                    optimizer.zero_grad()         #不同batch之间梯度不累加
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs=model(inputs)
                        preds = sigmoid(outputs)>0.5
                        l2_reg=torch.tensor([0],dtype=torch.float32)
                        l2_reg=l2_reg.to(self.device)
                        for param in model.parameters():
                            l2_reg+=torch.norm(param,2)
                        loss=criterion(outputs,labels)+0.2*l2_reg
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        running_loss+=loss.item()*inputs.size(0)
                        running_corrects+=torch.sum(preds == labels.byte()).detach().cpu().numpy()
                epoch_loss=running_loss/self.dataset_size[phase]
                epoch_acc=running_corrects/self.dataset_size[phase]
                log_file.write('{}   {}   '.format(epoch_loss,epoch_acc))
                print('{} Loss : {:.4f} Acc : {:.4f}'.format(phase,epoch_loss,epoch_acc))
                if phase=='val' and epoch_acc > best_acc:
                    best_acc=epoch_acc
                    best_model=deepcopy(model.state_dict())
                if epoch %10 ==0 :
                    torch.save(best_model,self.model_save_path+'resnet18_checkpoint.pth')
            log_file.write('\n')
            log_file.flush()
        time_elapsed=time.time()-start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc)) 
        torch.save(best_model,self.model_save_path+'PDMS_resnet18_100_alpha_0.2.pth')
        return(model)

if __name__ == '__main__' :
    data_path='/state/heyang/ZHEDA_PROJECT_DATA/datas/pdms_data/train_data/resnet_input_data/'
    save_path='/state/heyang/ZHEDA_PROJECT_DATA/models/pdms_resnet_model_alpha_0.2/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    a=resnet_model(data_path,200,save_path) 
    a.train_model(100)
