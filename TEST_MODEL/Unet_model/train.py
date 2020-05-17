import sys
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from .dice_loss import *

class train_model():
    def __init__(self,net_model,train_dataset,test_dataset,lr,trian_batch_size,test_batch_size,gpu,model_save_path,sample_save_path):
        self.net_model=net_model
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.lr=lr
        self.trian_batch_size=trian_batch_size
        self.test_batch_size=test_batch_size
        self.gpu=gpu
        self.model_save_path=model_save_path
        self.sample_save_path=sample_save_path
    def __call__(self,epoch_num):
        self.train_model(net_model=self.net_model,
                         epoch_num=epoch_num,
                         batch_size=self.trian_batch_size,
                         test_batch_size=self.test_batch_size,
                         lr=self.lr,
                         train_dataset=self.train_dataset,
                         test_dataset=self.test_dataset,
                         gpu=self.gpu,
                         model_save_path=self.model_save_path,
                         pic_save_path=self.sample_save_path,
                         check_bound=50,
                         log_path='./train_log')
        print('Train_complete')
    def tensor_to_pic(self,tensor,pic_type='RGB'):
        tensor=tensor.cpu()>0.5
        pil_img=transforms.ToPILImage()(tensor.float()).convert(pic_type)
        return(pil_img)
    def test_model(self,net_model,test_dataset,test_batch_size,sample_save_path,iter_number,save_sample,gpu,check_bound=20,brive_model=False):
        test_generator=test_dataset.get_batch()
        total_dice_acc=0
        test_batch_number=test_dataset.show_data_number()//test_batch_size
        for i in tqdm(range(test_batch_number)):
            x_test_img,y_test_img=next(test_generator)
            if brive_model and i%10!=0:
                continue
            if gpu:
                x_test_img=x_test_img.cuda()
                y_test_img=y_test_img.cuda()
            y_test_img_=net_model(x_test_img)
            if save_sample and i < check_bound:
                check_x=x_test_img.clone().cpu()[0]
                check_y=y_test_img.clone().cpu()[0]
                check_y_=y_test_img_.clone().cpu()[0]          
                x_img=self.tensor_to_pic(check_x)
                y_img=self.tensor_to_pic(check_y)
                y_img_=self.tensor_to_pic(check_y_)
                y_img_=Image.fromarray(np.asarray(y_img_))
                x_img.save('{}/{}_data.png'.format(sample_save_path,i))
                y_img.save('{}/{}_label.png'.format(sample_save_path,i))
                y_img_.save('{}/{}_pred.png'.format(sample_save_path,i)) 
            total_dice_acc+=MulticlassDiceAcc(y_test_img_,y_test_img).item()
            ave_dice_acc=total_dice_acc/test_batch_number
            if brive_model:
                ave_dice_acc=ave_dice_acc*10
        return(ave_dice_acc)
    def train_model(self,net_model,epoch_num,batch_size,test_batch_size,lr,train_dataset,test_dataset,gpu,model_save_path='./save_model/',pic_save_path='./save_pic/',check_bound=50,log_path='./train_log'):
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        if not os.path.exists(pic_save_path):
            os.makedirs(pic_save_path)
        # print info about dataset
        print('='*40)
        print('EPOCH_NUMBER    :{}'.format(epoch_num))
        print('BATCH_SIZE      :{}'.format(batch_size))
        print('TRAIN_DATA_NUM  :{}'.format(train_dataset.show_data_number()))
        print('TEST_DATA_NUM   :{}'.format(test_dataset.show_data_number()))
        print('USE_GPU         :{}'.format(gpu))
        net_model=nn.DataParallel(net_model.cuda(),device_ids=[0,1,2,3])
        print('='*40)
        log_file=open(log_path,'w')
        # build loss fun & optimizer
        criterion=nn.BCELoss()
        optimizer=optim.SGD(net_model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
        # save best model
        best_model=None
        best_acc=0
        best_epoch=None
        # build train and test generator
        train_generator=train_dataset.get_batch()
        # some stuff about test
        total_dice_acc=0
        # run epoch/batch
        for epoch in range(epoch_num):
            print('START EPOCH : {}/{}'.format(epoch,epoch_num))
            epoch_loss=0
            print('Start train')
            trian_batch_number=train_dataset.show_data_number()//batch_size
            for index in tqdm(range(trian_batch_number)):
                x_img, y_img=next(train_generator)
                if gpu:
                    x_img=x_img.cuda()
                    y_img=y_img.cuda()
                y_img_=net_model(x_img)
                flat_y_img_=y_img_.view(-1)
                flat_y_img=y_img.view(-1)
                batch_loss=criterion(flat_y_img_,flat_y_img)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                epoch_loss+=batch_loss.item()

                iter_number=epoch*trian_batch_number+index
                if iter_number % 100 == 0:
                    sample_save_path='{}/{}'.format(pic_save_path,iter_number)
                    if not os.path.exists(sample_save_path):
                        os.makedirs(sample_save_path)
                    ave_dice_acc = self.test_model(net_model,
                                                   test_dataset,
                                                   test_batch_size,
                                                   sample_save_path,
                                                   iter_number=iter_number,
                                                   save_sample=True,
                                                   gpu= True,
                                                   brive_model=True)
                    log_file.write('{}   {}   {}\n'.format(iter_number,batch_loss.item(),ave_dice_acc))
                    log_file.flush()
                    if ave_dice_acc>=best_acc:
                        best_acc=ave_dice_acc
                        best_model=deepcopy(net_model.state_dict())
                        best_epoch=iter_number

            ave_dice_acc = self.test_model(net_model,
                                           test_dataset,
                                           test_batch_size,
                                           sample_save_path,
                                           iter_number=epoch*batch_size+index,
                                           save_sample=False,
                                           gpu= True)
            ave_train_loss=epoch_loss*batch_size/trian_batch_number
            print('='*45)
            print('Epoch number : {}'.format(epoch))
            print('Train Loss   : {}'.format(ave_train_loss))
            print('Dice_Acc    : {}'.format(ave_dice_acc))
            print('='*45)
            if epoch %5 == 0 :
                torch.save(best_model,'{}/{}_checkpoint.pth'.format(model_save_path,epoch))

        print('='*45)
        print('Epoch Fisihed  : ')
        print('Best Val Epoch: {}'.format(best_epoch))
        print('Best Val Loss : {}'.format(best_acc))
        log_file.write('Best epoch :{} ;  Best val acc :  {}\n'.format(best_epoch,best_acc))
        log_file.close()
        torch.save(best_model,'{}/Unet_iter_{}.pth'.format(model_save_path,best_epoch))

