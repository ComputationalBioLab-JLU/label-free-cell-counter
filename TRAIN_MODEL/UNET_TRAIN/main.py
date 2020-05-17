#from Unet_model import Unet
import configparser
from Data_structure import dataset
from Unet_model import unet,train_model
import torch.nn as nn
import torch
from torchsummary import summary

cf=configparser.ConfigParser()
cf.read('./config')
TRAIN_PATH=cf.get('path','TRAIN_DATA_PATH')
TEST_PATH=cf.get('path','TEST_DATA_PATH')
X_SUFFIX=cf.get('path','X_SUFFIX')
Y_SUFFIX=cf.get('path','Y_SUFFIX')
BATCH_SIZE=int(cf.get('net_param','BATCH_SIZE'))
TEST_BATCH_SIZE=int(cf.get('net_param','TEST_BATCH_SIZE'))
CHANNELS=int(cf.get('net_param','CHANNELS'))
CLASSES=int(cf.get('net_param','CLASSES'))
GPU=bool(cf.get('net_param','GPU'))
EPOCHS=int(cf.get('net_param','EPOCHS'))
LR=float(cf.get('net_param','LR'))
TEST_SAVE_PATH=cf.get('path','TEST_PIC_SAVE_PATH')
MODEL_SAVE_PATH=cf.get('path','MODEL_SAVE_PATH')

train_dataset=dataset(TRAIN_PATH,X_SUFFIX,Y_SUFFIX,BATCH_SIZE,shuffle=True)
test_dataset=dataset(TEST_PATH,X_SUFFIX,Y_SUFFIX,TEST_BATCH_SIZE,shuffle=False)

def load_unet_model(CHANNELS,CLASSES,model_path):
    model=unet(CHANNELS,CLASSES)
    model=model.cuda()
    model.load_state_dict(torch.load(model_path),False)
    for param in model.parameters():
        param.requires_grad=True
    return(model)
unet_model=load_unet_model(CHANNELS,CLASSES,model_path='/state/heyang/OLD_CODE/cell_recognition_project/train_part/Unet_part/unet_model/Unet_iter_74000.pth')
#unet_model=unet(CHANNELS,CLASSES)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet_model = unet_model.to(device)
summary(unet_model,(3,572,572))
a=train_model(net_model=unet_model,
              train_dataset=train_dataset,
              test_dataset=test_dataset,
              lr=LR,
              trian_batch_size=BATCH_SIZE,
              test_batch_size=TEST_BATCH_SIZE,
              gpu=GPU,
              model_save_path=MODEL_SAVE_PATH,
              sample_save_path=TEST_SAVE_PATH)
a(EPOCHS)

