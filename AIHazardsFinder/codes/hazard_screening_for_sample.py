# -*- coding: utf-8 -*-
# Created by: ctt@2022
# chentiantian@dicp.ac.cn

import pandas as pd
import numpy as np
import math
import tensorflow as tf
from pyteomics import mgf
from utils import mgf_processing
import os

starMass = 50
endMass = 800
bin_width = 0.01
starNeu = 12
endNeu = 200
ms2_error=0.01


pred_dir=r"E:\第二个工作文章所用数据\样本数据\猪肉筛查结果"
model = tf.keras.models.load_model(r'选择合适的模型/classification_model.h5')
print(model.summary())
classes=np.load('神经网络-输入向量优化/classes_0.01.npy')
remove_list=[]
retain_list = []
result_df = pd.DataFrame()
row=0
os.chdir(pred_dir)
out_name='zhu_peak_addms2.mgf'
mgf_file_total = mgf.read(out_name)
shapeNum=len(mgf_file_total)
bin_num = round((endMass - starMass) / bin_width)
bin_neu_num = round((endNeu - starNeu) / bin_width)
for i in range(shapeNum):
    print(i)
    mgf_precursormz = round(float(mgf_file_total[i]['params']['pepmass'][0]), 5)
    scan = mgf_file_total[i]['params']['title'][-10:-1].split('=')[-1].split('"')[0]
    mgf_rt = round(float(mgf_file_total[i]['params']['rtinseconds'] / 60), 2)
    file_ori = mgf_file_total[i]['params']['title'].split(':')[1].split(',')[0].split('.')[0].strip('"')
    spectral = mgf_file_total[i]['m/z array']
    spectral_inten = mgf_file_total[i]['intensity array']
    spectral, spectral_inten, spectral_neu, spectral_neu_inten = mgf_processing.screening_vector_binning(
        spectral, spectral_inten, endMass, starMass, starNeu, endNeu, ms2_error, mgf_precursormz)
    result_df.loc[i, 'FILES'] = file_ori
    result_df.loc[i, 'SCAN'] = int(scan)
    result_df.loc[i, 'PEPMASS'] = mgf_precursormz
    result_df.loc[i, 'RT'] = mgf_rt
    ms2=''
    for m in range(len(spectral)):
        ms2 = ms2 + str(round(spectral[m], 5)) + ' ' + str(round(spectral_inten[m],3)) + ';'
    ms2=ms2.strip(';')
    result_df.loc[i, 'MSMS'] = ms2
    if len(spectral) >= 5:
        spectral_box = [0] * bin_num
        neu_box = [0] * bin_neu_num
        for l in range(len(spectral)):
            bin_position = math.floor((float(spectral[l]) - starMass) / bin_width)
            spectral_box[bin_position] = max(round(float(spectral_inten[l]), 3), spectral_box[bin_position])
        for l in range(len(spectral_neu)):
            bin_neu_position = math.floor((float(spectral_neu[l]) - starNeu) / bin_width)
            neu_box[bin_neu_position] = max(round(float(spectral_neu_inten[l]), 3), neu_box[bin_neu_position])
        screen_input=spectral_box+neu_box
        screen_input=np.array(screen_input).reshape(1, int(bin_num+bin_neu_num))
        pred_result = model.predict(screen_input)
        result_df.loc[i, 'pred_probability'] = max(pred_result[0])
        pred_result = np.argmax(pred_result, axis=-1)
        result_df.loc[i,'pred_category']=int(classes[pred_result[0]])+1
out_name=r'C:\Users\WIN10\PycharmProjects\2022_2_classfire\2022_python\AIHazard_Screening\选择合适的模型\0901_3_1.xlsx'
result_df.to_excel(out_name,index=False)

