# label-free-cell-counter
This is code for Label-free recognition and auto-counting of co-cultured cells using machine learning ,This result is from collaboration between MOE Key laboratory of Molecular Enzymology and Engineering ,school of life science of Jilin University and MOE Key Laboratory of Macromolecule Synthesis and Functionalization, Department of Polymer Science and Engineering of Zhejiang University. 

## Table of Contents
- [Background](#Background)
- [Usage](#Usage)
   - [Data Prefix](#Data_Prefix)
   - [Prepare_Data](#Prepare_Data)
   - [Training Part](#Training_Part)
      - [Unet Train](#Unet_Train)
      - [Resnet Train](#Resnet_Train)
   - [Count label part](#Count_label_part)
   - [Test Part](#Test_Part)
- [Maintainers](#Maintainers)
## Background
This project can detect and count co-cultured cells. Cell recognition and counting, especially in experiments concerning multiple cell types always require complicated dyeing steps and manual counting protocol. This project presents a label-free method to locate and tag each single cell in phase contrast images.    
![ ](https://github.com/ZaraYang/label-free-cell-counter/blob/master/schematic%20diagram/sample_img.png)  
This readme will explain things about:   
1.	Introduce the function of each part in this project.
2.	Use provided model to run inference in our dataset.
3.	Retrain model in new dataset.

## Usage
This part requires python3 and depend pytorch,sklearn,Open-CV and Numpy.  
### Data_Prefix
In our code, we will put different kind image into one folder, and use prefix to distinguish different kinds of pictures.  
They are shown in the table below ：  
  
| prefix  | Image type|
| ---------- | -----------|
| P   | bright field image   |
| RGB   | fluorescent image   |
| R   | Red channel of fluorescent image(after binarization)    |
| CR   | Red channel of fluorescent image  |
| B   | Blue channel of fluorescent image   |
| CB   | Blue channel of fluorescent image(after binarization)   |
| G   | Green channel of fluorescent image   |
| CG   | Green channel of fluorescent image(after binarization)   |
### Prepare_Data
This part includes some data pretreatment code.  
They are shown in the table below ：  
| file_name (.py) | code function|
| ---------- | -----------|
| make_pkl   | Save all tif image to pkl file    |
| split_RGB   | Split pkl file acorrding different channel  |
| makedirs   | Divide dataset into training set and test set and create floders for datasets|
| make_unet_data   | Cut the big picture into small pictures which is suitable for training, form the training set and test set, and extract the verification set from the training set   |
| make_resnet_data   | Form resnet training set, test set, and varification set from the dataset that after cluster. |  


These files can be called in turn. They will split big image into usable dataset for training model.  
### Training_Part
This part contains training code of Unet and Resnet model.  
#### Unet_Train
Unet's plays a role that can segment the shape of the cell nucleus from the brightfield picture in our project. When we train Unet model, the dataset should contain brightfield image as input data and nucleus fluorescent images as mask.  
Parameters of Unet Model is in the config file.  
    
    ./TRAIN_PART/UNET_MODEL/config  
     
In this config file, you can change path of dataset, learning rate, batch size, train epoch number and channel number of data.   
Then you can train your model by :   
    
    python3 ./TRAIN_PART/UNET_MODEL/main.py  
    
#### Resnet_Train
Resnet model can identify cells’ type according to the brightfield image of each cell.  We located each cell by performing DBSCAN cluster in nucleus fluorescent images, then chose window around the cluster center as train set.   
The code of Resnet model can be found in this file:  

    ./TRAIN_PART/RESNET_MODEL/train_resnet.py   

Resnet18 model that comes with pytorch will be loaded in code.  
The renset model can be trained de nove or making fine-tuning by loading resnet18 model which is already finished training.  

### Test_Part
This part can test the whole model's performance, including Unet model, DBSCAN cluster and Resnet model.  
Every parameter is in the config file :  

    ./label-free-cell-counter/TEST_MODEL/pred_config  
    
The config file includes models path, DBSCAN cluster's parameters and path of test dataset.
The test program can be run by following command:  

    python3 ./label-free-cell-counter/TEST_MODEL/main.py  

If models are trained well, predict result will be showen by enter cell brightfield image.  
### Count_label_part
This part is an auto-counting machine, which uses fluorescent images to locate and count each cell. We preform DBSCAN on nucleus fluorescent images to locate every cell, meanwhile, we compare the color of the nuclear region in the fluorescence image.  
All parameters can be changed in compute_label.py, which includes parameters of DBSCAN, and we set a low bound in it to filter noise.  
You can alse run auto-counting program by :  
 
    python3 ./label-free-cell-counter/COUNT_LABEL/compute_label.py  


## Maintainers
[@ZaraYang](https://github.com/ZaraYang).
