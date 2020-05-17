import os
import json
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

for i in range(2):
    x_list = []
    y_list = []
    x_all_list = []
    y_all_list =[]
        
    pred_result_path = '/state/heyang/cell_recognition_project/cell_count_machine/result/count_sample_label.js'
    label_result_path = '/state/heyang/zheda_project_history/zheda_cell_project/Old_data_retrain/data/train_unet_data/test/sampling_test_data_label/sample_label.js'
    label_dict = json.load(open(label_result_path,'r'))
    pred_dict = json.load(open(pred_result_path,'r'))   
    for key in tqdm(label_dict):
        x_list.append(label_dict[key][i])
        y_list.append(pred_dict[key][i])
        x_all_list.append(np.sum(label_dict[key]))
        y_all_list.append(np.sum(pred_dict[key]))
        if (abs(pred_dict[key][i] - label_dict[key][i]) >10 ):
            print(key)
    
    plt.plot(range(100),range(100),linewidth=2)
    plt.scatter(x_list, y_list, s=20)
    plt.xlim(0,60)
    plt.xlabel("label")
    plt.ylim(0,60)
    plt.ylabel("pred")
    plt.title("count acc in cell {}".format(i))
    plt.savefig("/state/heyang/cell_recognition_project/cell_count_machine/pic_result/count_sample_cell{}.jpg".format(i))
    plt.close()

plt.plot(range(100),range(100),linewidth=2)
plt.scatter(x_all_list, y_all_list, s=20)
plt.xlim(0,max(x_all_list))
plt.xlabel("label")
plt.ylim(0,max(x_all_list))
plt.ylabel("pred")
plt.title("count acc in all")
plt.savefig("/state/heyang/cell_recognition_project/cell_count_machine/pic_result/count_sample_all.jpg")
plt.close()
