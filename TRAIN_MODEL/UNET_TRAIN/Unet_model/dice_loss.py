import torch
import torch.nn as nn
from copy import deepcopy 
 
def dice_acc(input,target,smooth = 1):
	N = target.size(0)
	total_acc=0
	for i in range(N):
		temp_input=input[i,:,:]
		temp_target=target[i,:,:]
		temp_input=temp_input.cpu().detach().numpy()
		temp_target=temp_target.cpu().detach().numpy()
		temp_input = temp_input>0.5
		temp_target = temp_target>0.5
		intersection = temp_input * temp_target
		acc = (2 * intersection.sum()+ smooth) / (temp_input.sum() + temp_target.sum() + smooth)
		total_acc+=acc/N
	return(total_acc)
def MulticlassDiceAcc(input, target):
	C = target.shape[1]
	totalacc = 0
	for i in range(C):
		diceacc = dice_acc(input[:,i,:,:], target[:,i,:,:])
		totalacc += diceacc / C
	return (totalacc)	
