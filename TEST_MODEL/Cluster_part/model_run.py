import os
import numpy as np
import configparser
from glob import glob
from tqdm import tqdm


from Cluster_part.cluster_model import *

def cluster_model(config_file_path,pic_path,mask_path,windos_save_path,result_save_path):
    conf=configparser.ConfigParser()
    conf.read(config_file_path)

    DBSCAN_EPS=float(conf.get('Cluster_parameter','DBSCAN_eps'))
    DBSCAN_MIN_SAMPLES=float(conf.get('Cluster_parameter','DBSCAN_min_samples'))
    MEANSHIFT_BANDWIDTH=float(conf.get('Cluster_parameter','Meanshift_bandwidth'))
    MASK_PATH=mask_path#conf.get('Cluster_parameter','Cluster_data_path')
    PIC_PATH=pic_path#conf.get('Cluster_parameter','Cluster_data_path')
    MASK_SUFFIX=conf.get('suffix','Unet_pred_suffix')
    PIC_SUFFIX=conf.get('suffix','Pic_suffix')
    RESULT_SAVE_PATH=result_save_path#conf.get('Cluster_parameter','Cluster_result_path')
    WINDOW_SAVE_PATH=window_save_path#conf.get('Cluster_parameter','Cluster_window_path')
    LOW_BOUND=int(conf.get('Cluster_parameter','Low_bound'))
    UP_BOUND=int(conf.get('Cluster_parameter','Up_bound'))
    CIRCLE_COLOR=(0,0,255)
    CIRCLE_SIZE=20
    CUT_SIZE=224

    B_suffix=conf.get('suffix','B_suffix')
    CB_suffix=conf.get('suffix','CB_suffix')
    BGR_suffix=conf.get('suffix','BGR_suffix')

    DRAW_LIST=[B_suffix,CB_suffix,PIC_SUFFIX]

    model=cluster_model(DBSCAN_EPS,DBSCAN_MIN_SAMPLES,MEANSHIFT_BANDWIDTH,CIRCLE_COLOR,CIRCLE_SIZE,LOW_BOUND,UP_BOUND)
    for mask_path in tqdm(glob('{}/{}*'.format(MASK_PATH,MASK_SUFFIX))):
        pic_path=mask_path.replace(MASK_PATH,PIC_PATH).replace(MASK_SUFFIX,PIC_SUFFIX)
        model(mask_path,pic_path,MASK_SUFFIX,PIC_SUFFIX,WINDOW_SAVE_PATH,RESULT_SAVE_PATH,CUT_SIZE,DRAW_LIST) 
