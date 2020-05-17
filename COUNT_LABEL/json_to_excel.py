import json
import xlwt
import os

label_json_path_list  =['/state/heyang/cell_recognition_project/cell_count_machine/result/OLD_label.js',
                        '/state/heyang/cell_recognition_project/cell_count_machine/result/trans_TCPS_label.js',
                        '/state/heyang/cell_recognition_project/cell_count_machine/result/trans_PDMS_label.js',
                        '/state/heyang/zheda_project_history/zheda_cell_project/Old_data_retrain/data/train_unet_data/test/sampling_test_data_label/sample_label.js'] 
pred_json_path_list   =['/state/heyang/cell_recognition_project/pred_part/code/pred_result/old_data_pred.js',
                        '/state/heyang/cell_recognition_project/pred_part/code/pred_result/TCPS_pred.js',
                        '/state/heyang/cell_recognition_project/pred_part/code/pred_result/PDMS_pred.js',
                        '/state/heyang/cell_recognition_project/cell_count_machine/result/count_sample_label.js']



def load_data(book,sheet_name,label_path,pred_path):
    sheet = book.add_sheet(sheet_name)
    label_data = json.load(open(label_path,'r'))
    pred_data = json.load(open(pred_path,'r'))

    sheet.write(0,0,'img_id')
    sheet.write(0,1,'label_0')
    sheet.write(0,2,'label_1')
    sheet.write(0,3,'pred_0')
    sheet.write(0,4,'pred_1')

    index = 1
    for key in label_data:
        sheet.write(index,0,key)
        sheet.write(index,1,label_data[key][0])
        sheet.write(index,2,label_data[key][1])
        sheet.write(index,3,pred_data[key][0])
        sheet.write(index,4,pred_data[key][1])
        index += 1

book = xlwt.Workbook()
for i in range(len(label_json_path_list)):
    label_path = label_json_path_list[i]
    pred_path = pred_json_path_list[i]
    name = os.path.basename(label_path)
    load_data(book,name,label_path,pred_path)

book.save('./data_excel.xls')

