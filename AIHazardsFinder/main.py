# -*- coding: utf-8 -*-
# Created by: ctt@2022
# chentiantian@dicp.ac.cn

import os
import sys
import json
import pandas as pd
import numpy as np
import math
from AIHazardsFinder import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5.Qt import QThread
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtWidgets import QMenu
from PyQt5.QtWidgets import QDesktopWidget
import pyisopach
from collections import Counter
from keras import models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pyteomics import mzxml
from pyteomics import mgf
import mgf_processing


starMass = 50
endMass = 800
bin_width = 0.01
starNeu = 12
endNeu = 200
ms2_error = 0.01
ms1_error = 10

class DrawFunction(object):
    def Image_In_Process(self, feature):
        mzxml_file = mzxml.read(feature.mzxmlpath)
        inten = []
        rt = []
        for spec in mzxml_file:
            if spec['msLevel'] == 1:
                if int(feature.rt_max) > 120:
                    mzxml_rt = round((float(spec['retentionTime']) / 60), 2)
                else:
                    mzxml_rt = round(float(spec['retentionTime']), 2)
                mzxml_ms = spec['m/z array']
                mzxml_ms_inten = spec['intensity array']
                for l in range(len(mzxml_ms)):
                    if (10 ** 6) * abs(feature.ms1_show[(int(feature.index_no) - 1)] - mzxml_ms[l]) / mzxml_ms[l] < ms1_error:
                        inten.append(mzxml_ms_inten[l])
                        rt.append(mzxml_rt)
                        break
        nl_max='%.3g'%max(inten)
        inten = list(inten / max(inten) * 100)
        index_max = inten.index(max(inten))
        peak_rt = round(rt[index_max], 2)
        feature.label_2.setText(
            ('Extracted Ion Chromatogram' + '     MS1: ' + str(
                feature.ms1_show[(int(feature.index_no) - 1)]) +'   NL: '+ str(nl_max)+ '   RT: ' + str(
                peak_rt)))
        feature.fig1.clear()
        ax1 = feature.fig1.add_subplot(111)
        # 调整图像大小
        ax1.cla()
        rt_max_now = max(rt)
        rt_min_now = min(rt)
        if rt_max_now < (feature.rt_max - 2):
            x_rt1 = rt_max_now + 2
        if rt_max_now >= (feature.rt_max - 2):
            x_rt1 = feature.rt_max
        if rt_min_now - 2 < 0:
            x_rt0 = 0
        if rt_min_now - 2 >= 0:
            x_rt0 = rt_min_now - 2
        for x in range(1, (int((x_rt1 - rt_max_now) / 0.1) + 1)):
            rt.append(rt_max_now + 0.1 * x)
            inten.append(0)
        for x in range(1, (int((rt_min_now - x_rt0) / 0.1) + 1)):
            rt.insert(0, rt_min_now - 0.1 * x)
            inten.insert(0, 0)
        ax1.plot(rt, inten, linewidth=1, color='black')
        ax1.set_ylim((0, 110))
        ax1.set_xlim((x_rt0, x_rt1))
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Relative Intensity')
        feature.fig1.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=None, hspace=None)
        feature.canvas1.draw()
        ms2_show_1 = feature.ms2_show[int((int(feature.index_no) - 1))].split(';')
        infos_count1 = len(ms2_show_1)
        msms = []
        msms_inten = []
        for l in range(infos_count1):
            msms.append(float((ms2_show_1[l].split(' ')[0])))
            msms_inten.append((round(float(ms2_show_1[l].split(' ')[1]) * 100, 1)))

        feature.fig2.clear()
        ax2 = feature.fig2.add_subplot(111)
        # 调整图像大小
        ax2.cla()
        msms_max = max(msms)
        ax2.bar(msms, msms_inten, width=msms_max / 500, color='black')
        index_max = [x[0] for x in sorted(enumerate(msms_inten), key=lambda x: x[1])[-5:]]
        for y, x in enumerate(msms):
            if y in index_max:
                plt.text(x, msms_inten[y] + 0.5, "%s" % round(x, 4), ha="center", va="bottom")
        ax2.set_ylim((0, 110))
        ax2.set_xlabel('mz')
        ax2.set_ylabel('Relative Intensity')
        feature.fig2.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.95, wspace=None, hspace=None)
        feature.canvas2.draw()

    def Image_In_ProcessII(self, feature):
        feature.fig1.clear()
        feature.canvas1.draw()
        feature.fig2.clear()
        feature.label_2.setText('')
        feature.canvas2.draw()

class MyThreadOne(QThread):
    classification_signal = pyqtSignal(str)
    show_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.input_path = ''

    def run(self):
        model = models.load_model('supports/classification_model.h5')
        classes = np.load('supports/classes.npy')
        input_path_dict = json.loads(self.input_path)
        mzxmlpath = input_path_dict.get('mzxmlpath')
        peaklistpath = input_path_dict.get('peaklistpath')
        peakList = pd.read_excel(peaklistpath)
        peakMS1 = peakList['mz']
        peakRT = peakList['rt[min]']
        AIClassification = pd.DataFrame()
        bin_num = round((endMass - starMass) / bin_width)
        bin_neu_num = round((endNeu - starNeu) / bin_width)
        num = -1
        mzxml_file = mzxml.read(mzxmlpath)
        rt_max = 0
        scan_total = 0
        for spec in mzxml_file:
            if spec['msLevel'] == 2:
                scan_total = scan_total + 1
                rt0 = round(float(spec['retentionTime']), 2)
                rt_max = max(rt_max, rt0)
        mzxml_file = mzxml.read(mzxmlpath)
        scan_index = 0
        for spec in mzxml_file:
            scan_index = scan_index + 1
            if (scan_index % 50) == 0:
                percentage = int((scan_index / scan_total) * 100)
                if percentage <= 99:
                    self.classification_signal.emit(json.dumps(percentage))
                if percentage > 99:
                    self.classification_signal.emit(json.dumps(99))
            if spec['msLevel'] == 2:
                num = num + 1
                file_ori = str(mzxmlpath.split('/')[-1].split('.')[-2])
                if int(rt_max) > 120:
                    precursor_rt = round((float(spec['retentionTime']) / 60), 2)
                else:
                    precursor_rt = round(float(spec['retentionTime']), 2)
                precursor = round(
                    float(spec['precursorMz'][0]['precursorMz']), 4)
                scan = spec['num']
                spectral = spec['m/z array']
                spectral_inten = spec['intensity array']
                spectral_inten = spectral_inten / max(spectral_inten)
                keep = np.where(spectral_inten > 0.01)
                spectral = spectral[keep]
                spectral_inten = spectral_inten[keep]
                ms2 = ''
                for m in range(len(spectral)):
                    ms2 = ms2 + str(round(spectral[m], 4)) + ' ' + str(round(spectral_inten[m], 3)) + ';'
                ms2 = ms2.strip(';')
                spectral, spectral_inten, spectral_neu, spectral_neu_inten = mgf_processing.screening_vector_binning(
                    spectral, spectral_inten, endMass, starMass, starNeu, endNeu, ms2_error, precursor)
                AIClassification.loc[num, 'File'] = file_ori
                AIClassification.loc[num, 'Scan'] = int(scan)
                AIClassification.loc[num, 'Precursor ion'] = precursor
                AIClassification.loc[num, 'RT'] = precursor_rt
                AIClassification.loc[num, 'MSMS'] = ms2
                if len(spectral) >= 5:
                    spectral_box = [0] * bin_num
                    neu_box = [0] * bin_neu_num
                    for l in range(len(spectral)):
                        bin_position = math.floor((float(spectral[l]) - starMass) / bin_width)
                        spectral_box[bin_position] = max(round(float(spectral_inten[l]), 3),
                                                         spectral_box[bin_position])
                    for l in range(len(spectral_neu)):
                        bin_neu_position = math.floor((float(spectral_neu[l]) - starNeu) / bin_width)
                        neu_box[bin_neu_position] = max(round(float(spectral_neu_inten[l]), 3),
                                                        neu_box[bin_neu_position])
                    screen_input = spectral_box + neu_box
                    screen_input = np.array(screen_input).reshape(1, int(bin_num + bin_neu_num))
                    pred_result = model.predict(screen_input)
                    pred_class = np.argmax(pred_result, axis=-1)
                    if int(pred_class[0]) != 32:
                        for m in range(len(peakMS1)):
                            if 10 ** 6 * abs(precursor - peakMS1[m]) / precursor < 10 and abs(
                                    peakRT[m] - precursor_rt) < 0.2:
                                AIClassification.loc[num, 'Probability'] = round(max(pred_result[0]), 2)
                                AIClassification.loc[num, 'Predicted class'] = classes[pred_class[0]]
        AIClassification.replace('', np.nan, inplace=True)
        AIClassification.dropna(subset=['Predicted class'], axis=0, inplace=True)
        AIClassification_path = 'results/AIClassification_' + str(file_ori) + '.xlsx'
        file_show_info = {'AIClassification_path': AIClassification_path, 'rt_max': rt_max}
        AIClassification.to_excel(AIClassification_path, index=False)
        self.show_signal.emit((json.dumps(file_show_info)))
        self.classification_signal.emit(json.dumps(100))


class MyThreadOneII(QThread):
    classification_signal = pyqtSignal(str)
    show_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.input_path = ''

    def run(self):
        model = models.load_model('supports/classification_model.h5')
        classes = np.load('supports/classes.npy')
        input_path_dict = json.loads(self.input_path)
        mzxmlpath = input_path_dict.get('mzxmlpath')
        AIClassification = pd.DataFrame()
        bin_num = round((endMass - starMass) / bin_width)
        bin_neu_num = round((endNeu - starNeu) / bin_width)
        num = -1
        mzxml_file = mzxml.read(mzxmlpath)
        rt_max = 0
        scan_total = 0
        for spec in mzxml_file:
            if spec['msLevel'] == 2:
                scan_total = scan_total + 1
                rt0 = round(float(spec['retentionTime']), 2)
                rt_max = max(rt_max, rt0)
        mzxml_file = mzxml.read(mzxmlpath)
        scan_index = 0
        for spec in mzxml_file:
            scan_index = scan_index + 1
            if (scan_index % 50) == 0:
                percentage = int((scan_index / scan_total) * 100)
                if percentage <= 99:
                    self.classification_signal.emit(json.dumps(percentage))
                if percentage > 99:
                    self.classification_signal.emit(json.dumps(99))
            if spec['msLevel'] == 2:
                num = num + 1
                file_ori = str(mzxmlpath.split('/')[-1].split('.')[-2])
                if int(rt_max) > 120:
                    precursor_rt = round((float(spec['retentionTime']) / 60), 2)
                else:
                    precursor_rt = round(float(spec['retentionTime']), 2)
                precursor = round(
                    float(spec['precursorMz'][0]['precursorMz']), 4)
                scan = spec['num']
                spectral = spec['m/z array']
                spectral_inten = spec['intensity array']
                spectral_inten = spectral_inten / max(spectral_inten)
                keep = np.where(spectral_inten > 0.01)
                spectral = spectral[keep]
                spectral_inten = spectral_inten[keep]
                ms2 = ''
                for m in range(len(spectral)):
                    ms2 = ms2 + str(round(spectral[m], 4)) + ' ' + str(round(spectral_inten[m], 3)) + ';'
                ms2 = ms2.strip(';')
                spectral, spectral_inten, spectral_neu, spectral_neu_inten = mgf_processing.screening_vector_binning(
                    spectral, spectral_inten, endMass, starMass, starNeu, endNeu, ms2_error, precursor)
                AIClassification.loc[num, 'File'] = file_ori
                AIClassification.loc[num, 'Scan'] = int(scan)
                AIClassification.loc[num, 'Precursor ion'] = precursor
                AIClassification.loc[num, 'RT'] = precursor_rt
                AIClassification.loc[num, 'MSMS'] = ms2
                if len(spectral) >= 5:
                    spectral_box = [0] * bin_num
                    neu_box = [0] * bin_neu_num
                    for l in range(len(spectral)):
                        bin_position = math.floor((float(spectral[l]) - starMass) / bin_width)
                        spectral_box[bin_position] = max(round(float(spectral_inten[l]), 3),
                                                         spectral_box[bin_position])
                    for l in range(len(spectral_neu)):
                        bin_neu_position = math.floor((float(spectral_neu[l]) - starNeu) / bin_width)
                        neu_box[bin_neu_position] = max(round(float(spectral_neu_inten[l]), 3),
                                                        neu_box[bin_neu_position])
                    screen_input = spectral_box + neu_box
                    screen_input = np.array(screen_input).reshape(1, int(bin_num + bin_neu_num))
                    pred_result = model.predict(screen_input)
                    pred_class = np.argmax(pred_result, axis=-1)
                    if int(pred_class[0]) != 32:
                        AIClassification.loc[num, 'Probability'] = round(max(pred_result[0]), 2)
                        AIClassification.loc[num, 'Predicted class'] = classes[pred_class[0]]
        AIClassification.replace('', np.nan, inplace=True)
        AIClassification.dropna(subset=['Predicted class'], axis=0, inplace=True)
        AIClassification_path = 'results/AIClassification_' + str(file_ori) + '.xlsx'
        file_show_info = {'AIClassification_path': AIClassification_path, 'rt_max': rt_max}
        AIClassification.to_excel(AIClassification_path, index=False)
        self.show_signal.emit((json.dumps(file_show_info)))
        self.classification_signal.emit(json.dumps(100))


class MyThreadTwo(QThread):
    classification_signal = pyqtSignal(str)
    show_signal=pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.input_path = ''
    def run(self):
        model = models.load_model('supports/classification_model.h5')
        classes = np.load('supports/classes.npy')
        input_path_dict = json.loads(self.input_path)
        mgfpath = input_path_dict.get('mgfpath')
        peaklistpath = input_path_dict.get('peaklistpath')
        peakList = pd.read_excel(peaklistpath)
        peakMS1 = peakList['mz']
        peakRT = peakList['rt[min]']
        AIClassification = pd.DataFrame()
        bin_num = round((endMass - starMass) / bin_width)
        bin_neu_num = round((endNeu - starNeu) / bin_width)
        mgf_file = mgf.read(mgfpath)
        rt_max = 0
        for i in range(len(mgf_file)):
            rt0 = float(mgf_file[i]['params']['rtinseconds'])
            rt_max = max(rt_max, rt0)
        scan_total = len(mgf_file)
        scan_index = 0
        for i in range(len(mgf_file)):
            scan_index = scan_index + 1
            if (scan_index % 50) == 0:
                percentage = int((scan_index / scan_total) * 100)
                if percentage <= 99:
                    self.classification_signal.emit(json.dumps(percentage))
                if percentage > 99:
                    self.classification_signal.emit(json.dumps(99))
            file_ori = mgf_file[i]['params']['title'].split(':')[1].split(',')[0].strip('"').split('.raw')[0]
            if int(rt_max) > 120:
                precursor_rt = round(float(mgf_file[i]['params']['rtinseconds']/60),2)
            else:
                precursor_rt = round(float(mgf_file[i]['params']['rtinseconds']),2)
            precursor=round(float(mgf_file[i]['params']['pepmass'][0]), 4)
            scan= int(mgf_file[i]['params']['title'].split('=')[-1].strip('"'))
            spectral = mgf_file[i]['m/z array']
            spectral_inten = mgf_file[i]['intensity array']
            spectral_inten = spectral_inten / max(spectral_inten)
            keep = np.where(spectral_inten > 0.01)
            spectral = spectral[keep]
            spectral_inten = spectral_inten[keep]
            ms2 = ''
            for m in range(len(spectral)):
                ms2 = ms2 + str(round(spectral[m], 4)) + ' ' + str(round(spectral_inten[m], 3)) + ';'
            ms2 = ms2.strip(';')
            spectral, spectral_inten, spectral_neu, spectral_neu_inten = mgf_processing.screening_vector_binning(
                    spectral, spectral_inten, endMass, starMass, starNeu, endNeu, ms2_error, precursor)
            AIClassification.loc[i, 'File'] = file_ori
            AIClassification.loc[i, 'Scan'] = int(scan)
            AIClassification.loc[i, 'Precursor ion'] = precursor
            AIClassification.loc[i, 'RT'] = precursor_rt
            AIClassification.loc[i, 'MSMS'] = ms2
            if len(spectral) >= 5:
                spectral_box = [0] * bin_num
                neu_box = [0] * bin_neu_num
                for l in range(len(spectral)):
                    bin_position = math.floor((float(spectral[l]) - starMass) / bin_width)
                    spectral_box[bin_position] = max(round(float(spectral_inten[l]), 3),
                                                     spectral_box[bin_position])
                for l in range(len(spectral_neu)):
                    bin_neu_position = math.floor((float(spectral_neu[l]) - starNeu) / bin_width)
                    neu_box[bin_neu_position] = max(round(float(spectral_neu_inten[l]), 3),
                                                    neu_box[bin_neu_position])
                screen_input = spectral_box + neu_box
                screen_input = np.array(screen_input).reshape(1, int(bin_num + bin_neu_num))
                pred_result = model.predict(screen_input)
                pred_class = np.argmax(pred_result, axis=-1)
                if int(pred_class[0]) != 32:
                    for m in range(len(peakMS1)):
                        if 10 ** 6 * abs(precursor - peakMS1[m]) / precursor < 10 and abs(
                                peakRT[m] - precursor_rt) < 0.2:
                            AIClassification.loc[i, 'Probability'] = round(max(pred_result[0]), 2)
                            AIClassification.loc[i, 'Predicted class'] = classes[pred_class[0]]
        AIClassification.replace('', np.nan, inplace=True)
        AIClassification.dropna(subset=['Predicted class'], axis=0, inplace=True)
        AIClassification_path = 'results/AIClassification_' + str(mgfpath.split('/')[-1].split('.')[-2]) + '.xlsx'
        AIClassification.to_excel(AIClassification_path, index=False)
        file_show_info = {'AIClassification_path': AIClassification_path}
        self.show_signal.emit((json.dumps(file_show_info)))
        self.classification_signal.emit(json.dumps(100))


class MyThreadTwoII(QThread):
    classification_signal = pyqtSignal(str)
    show_signal=pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.input_path = ''

    def run(self):
        model = models.load_model('supports/classification_model.h5')
        classes = np.load('supports/classes.npy')
        input_path_dict = json.loads(self.input_path)
        mgfpath = input_path_dict.get('mgfpath')
        AIClassification = pd.DataFrame()
        bin_num = round((endMass - starMass) / bin_width)
        bin_neu_num = round((endNeu - starNeu) / bin_width)
        mgf_file = mgf.read(mgfpath)
        rt_max = 0
        for i in range(len(mgf_file)):
            rt0 = float(mgf_file[i]['params']['rtinseconds'])
            rt_max = max(rt_max, rt0)
        scan_total = len(mgf_file)
        scan_index = 0
        for i in range(len(mgf_file)):
            scan_index = scan_index + 1
            if (scan_index % 50) == 0:
                percentage = int((scan_index / scan_total) * 100)
                if percentage <= 99:
                    self.classification_signal.emit(json.dumps(percentage))
                if percentage > 99:
                    self.classification_signal.emit(json.dumps(99))
            file_ori = mgf_file[i]['params']['title'].split(':')[1].split(',')[0].strip('"').split('.raw')[0]
            if int(rt_max) > 120:
                precursor_rt = round(float(mgf_file[i]['params']['rtinseconds']/60),2)
            else:
                precursor_rt = round(float(mgf_file[i]['params']['rtinseconds']),2)
            precursor=round(float(mgf_file[i]['params']['pepmass'][0]), 4)
            scan= int(mgf_file[i]['params']['title'].split('=')[-1].strip('"'))
            spectral = mgf_file[i]['m/z array']
            spectral_inten = mgf_file[i]['intensity array']
            spectral_inten = spectral_inten / max(spectral_inten)
            keep = np.where(spectral_inten > 0.01)
            spectral = spectral[keep]
            spectral_inten = spectral_inten[keep]
            ms2 = ''
            for m in range(len(spectral)):
                ms2 = ms2 + str(round(spectral[m], 4)) + ' ' + str(round(spectral_inten[m], 3)) + ';'
            ms2 = ms2.strip(';')
            spectral, spectral_inten, spectral_neu, spectral_neu_inten = mgf_processing.screening_vector_binning(
                    spectral, spectral_inten, endMass, starMass, starNeu, endNeu, ms2_error, precursor)
            AIClassification.loc[i, 'File'] = file_ori
            AIClassification.loc[i, 'Scan'] = int(scan)
            AIClassification.loc[i, 'Precursor ion'] = precursor
            AIClassification.loc[i, 'RT'] = precursor_rt
            AIClassification.loc[i, 'MSMS'] = ms2
            if len(spectral) >= 5:
                spectral_box = [0] * bin_num
                neu_box = [0] * bin_neu_num
                for l in range(len(spectral)):
                    bin_position = math.floor((float(spectral[l]) - starMass) / bin_width)
                    spectral_box[bin_position] = max(round(float(spectral_inten[l]), 3),
                                                     spectral_box[bin_position])
                for l in range(len(spectral_neu)):
                    bin_neu_position = math.floor((float(spectral_neu[l]) - starNeu) / bin_width)
                    neu_box[bin_neu_position] = max(round(float(spectral_neu_inten[l]), 3),
                                                    neu_box[bin_neu_position])
                screen_input = spectral_box + neu_box
                screen_input = np.array(screen_input).reshape(1, int(bin_num + bin_neu_num))
                pred_result = model.predict(screen_input)
                pred_class = np.argmax(pred_result, axis=-1)
                if int(pred_class[0]) != 32:
                    AIClassification.loc[i, 'Probability'] = round(max(pred_result[0]), 2)
                    AIClassification.loc[i, 'Predicted class'] = classes[pred_class[0]]
        AIClassification.replace('', np.nan, inplace=True)
        AIClassification.dropna(subset=['Predicted class'], axis=0, inplace=True)
        AIClassification_path = 'results/AIClassification_' + str(mgfpath.split('/')[-1].split('.')[-2]) + '.xlsx'
        AIClassification.to_excel(AIClassification_path, index=False)
        file_show_info = {'AIClassification_path': AIClassification_path}
        self.show_signal.emit((json.dumps(file_show_info)))
        self.classification_signal.emit(json.dumps(100))


class MyThreadThree(QThread):
    classification_signal = pyqtSignal(str)
    show_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.input_path = ''

    def run(self):
        self.classification_signal.emit(json.dumps(0))
        input_path_dict = json.loads(self.input_path)
        eleConspath = input_path_dict.get('eleConspath')
        ClassificationFormula_path = input_path_dict.get('AIClassification_path')
        mzxmlpath = input_path_dict.get('mzxmlpath')
        rt_max=input_path_dict.get('rt_max')
        ClassificationFormula=pd.read_excel(ClassificationFormula_path)
        ClassificationFormula_class = ClassificationFormula['Predicted class']
        ClassificationFormula_MS1 = ClassificationFormula['Precursor ion']
        ClassificationFormula_rt=ClassificationFormula['RT']
        eleCon = pd.read_excel(eleConspath)
        eleCon_class = eleCon['Classes']

        for i in range(len(ClassificationFormula_MS1)):
            element_limit = []
            for l in range(len(eleCon_class)):
                if ClassificationFormula_class[i] == eleCon_class[l]:
                    for m in range(1, eleCon.shape[1]):
                        element_limit.append(eleCon.iloc[l, m])
                    exact_default = {'H': 1.007825, 'C': 12.000000, 'N': 14.003074, 'O': 15.994915, 'F': 18.998403,
                                     'P': 30.973763,
                                     'S': 31.972072, 'Cl': 34.968853, 'Br': 78.918336, 'I': 126.90447}
                    mz=ClassificationFormula_MS1[i]
                    formula_0 = []
                    for c_num in range(element_limit[0], element_limit[11] + 1):
                        for h_num in range(element_limit[1], element_limit[12] + 1):
                            for f_num in range(element_limit[2], element_limit[13] + 1):
                                for cl_num in range(element_limit[3], element_limit[14] + 1):
                                    for br_num in range(element_limit[4], element_limit[15] + 1):
                                        for i_num in range(element_limit[5], element_limit[16] + 1):
                                            for n_num in range(element_limit[6], element_limit[17] + 1):
                                                for o_num in range(element_limit[7], element_limit[18] + 1):
                                                    for p_num in range(element_limit[8], element_limit[19] + 1):
                                                        for s_num in range(element_limit[9], element_limit[20] + 1):
                                                            elements_nums = np.array([c_num, h_num, f_num, cl_num, br_num, i_num, n_num,o_num,p_num, s_num])
                                                            unsaturation = c_num + 1 - (h_num + f_num + cl_num + br_num + i_num - n_num) / 2
                                                            if 10 ** 6 * abs((c_num * exact_default.get('C') + h_num * exact_default.get('H') +o_num * exact_default.get(
                                                                        'O') + n_num * exact_default.get('N') + s_num * exact_default.get('S') + p_num * exact_default.get('P') +
                                                                              f_num * exact_default.get('F') + cl_num * exact_default.get(
                                                                        'Cl') + br_num * exact_default.get('Br') + i_num * exact_default.get('I')) - (float(mz) - 1.00728)) / float( mz) < ms1_error:
                                                                elements = np.array(['C', 'H', 'F', 'Cl', 'Br', 'I', 'N', 'O', 'P','S', ])
                                                                if element_limit[10] <= unsaturation <= element_limit[21]:
                                                                    if (h_num + f_num + cl_num + br_num + i_num - 2) / 2 < c_num and c_num < 2 * ( h_num + f_num + cl_num + br_num + i_num) + 2:
                                                                        ele_loc = np.where(elements_nums >= 1)
                                                                        formula = ''
                                                                        for l in ele_loc[0]:
                                                                            if elements_nums[l] == 1:
                                                                                formula = (str(formula) + str(elements[l]))
                                                                            else:
                                                                                formula = (str(formula) + str(elements[l]) + str(elements_nums[l]))
                                                                        mol = pyisopach.Molecule(formula)
                                                                        isotope = mol.isotopic_distribution()
                                                                        isotope1 = isotope[0][0] + 1.00728
                                                                        isotope2_inten = list(isotope[1][1:])
                                                                        isotope2 = list(isotope[0][1:])
                                                                        max_index = isotope2_inten.index(max(isotope2_inten))
                                                                        isotope2 = isotope2[max_index]
                                                                        isotope2_inten = isotope2_inten[max_index]
                                                                        isotope_fit_num = 0
                                                                        mzxml_file = mzxml.read(mzxmlpath)
                                                                        for spec in mzxml_file:
                                                                            if int(rt_max) > 120:
                                                                                mzxml_rt = round((float(spec['retentionTime']) / 60), 2)
                                                                            else:
                                                                                mzxml_rt = round(float(spec['retentionTime']), 2)
                                                                            if abs(mzxml_rt-ClassificationFormula_rt[i])< 0.2:
                                                                                if spec['msLevel'] == 1:
                                                                                    mzxml_ms = spec['m/z array']
                                                                                    mzxml_ms_inten = spec['intensity array']
                                                                                    for x in range(len(mzxml_ms)):
                                                                                        if (10 ** 6) * abs(isotope1 - mzxml_ms[x])/mzxml_ms[x] < ms1_error:
                                                                                            for m in range(x,len(mzxml_ms)):
                                                                                                if (10 ** 6) * abs(isotope2 + 1.00728 -mzxml_ms[m]) / mzxml_ms[m] < ms1_error:
                                                                                                    if abs((mzxml_ms_inten[m] /mzxml_ms_inten[x]) * 100 - isotope2_inten) / isotope2_inten < 0.3:
                                                                                                        isotope_fit_num = isotope_fit_num + 1
                                                                                                        break
                                                                        if isotope_fit_num > 0:
                                                                            formula_0.append(formula)

                    number = Counter(formula_0)
                    # 使用most_common()函数
                    result = number.most_common()
                    # 将结果打印出来
                    formula_final = ''
                    if len(result) >= 1:
                        for x in range(len(result)):
                            formula_final = formula_final + '; ' + result[x][0]
                    if len(result) == 0:
                        formula_final = ''
                    formula_final = formula_final.strip('; ')
                    ClassificationFormula.loc[i, 'Molecular formula'] = formula_final
                    percentage = int((i/len(ClassificationFormula_MS1)) * 100)
                    if percentage <= 99:
                        self.classification_signal.emit(json.dumps(percentage))
                    if percentage > 99:
                        self.classification_signal.emit(json.dumps(99))
        ClassificationFormula.replace('', np.nan, inplace=True)
        ClassificationFormula.dropna(subset=['Molecular formula'], axis=0, inplace=True)
        ClassificationFormula_path = 'results/AIClassification_FormulaFiltration_' + str(mzxmlpath.split('/')[-1].split('.')[-2]) + '.xlsx'
        ClassificationFormula.to_excel(ClassificationFormula_path,index=False)
        file_show_info = {'AIClassification_path': ClassificationFormula_path,'rt_max':rt_max}
        self.show_signal.emit((json.dumps(file_show_info)))
        self.classification_signal.emit(json.dumps(100))

class MyThreadFour(QThread):
    classification_signal = pyqtSignal(str)
    show_signal=pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.input_path = ''

    def run(self):
        self.classification_signal.emit(json.dumps(0))
        input_path_dict = json.loads(self.input_path)
        eleConspath = input_path_dict.get('eleConspath')
        ClassificationFormula_path = input_path_dict.get('AIClassification_path')
        mgfpath = input_path_dict.get('mgfpath')
        ClassificationFormula=pd.read_excel(ClassificationFormula_path)
        ClassificationFormula_class = ClassificationFormula['Predicted class']
        ClassificationFormula_MS1 = ClassificationFormula['Precursor ion']
        eleCon = pd.read_excel(eleConspath)
        eleCon_class = eleCon['Classes']
        for i in range(len(ClassificationFormula_MS1)):
            element_limit = []
            for l in range(len(eleCon_class)):
                if ClassificationFormula_class[i] == eleCon_class[l]:
                    for m in range(1, eleCon.shape[1]):
                        element_limit.append(eleCon.iloc[l, m])
                    exact_default = {'H': 1.007825, 'C': 12.000000, 'N': 14.003074, 'O': 15.994915, 'F': 18.998403,
                                     'P': 30.973763,
                                     'S': 31.972072, 'Cl': 34.968853, 'Br': 78.918336, 'I': 126.90447}
                    mz=ClassificationFormula_MS1[i]
                    formula_0 = []
                    for c_num in range(element_limit[0], element_limit[11] + 1):
                        for h_num in range(element_limit[1], element_limit[12] + 1):
                            for f_num in range(element_limit[2], element_limit[13] + 1):
                                for cl_num in range(element_limit[3], element_limit[14] + 1):
                                    for br_num in range(element_limit[4], element_limit[15] + 1):
                                        for i_num in range(element_limit[5], element_limit[16] + 1):
                                            for n_num in range(element_limit[6], element_limit[17] + 1):
                                                for o_num in range(element_limit[7], element_limit[18] + 1):
                                                    for p_num in range(element_limit[8], element_limit[19] + 1):
                                                        for s_num in range(element_limit[9], element_limit[20] + 1):
                                                            elements_nums = np.array([c_num, h_num, f_num, cl_num, br_num, i_num, n_num,o_num,p_num, s_num])
                                                            unsaturation = c_num + 1 - (h_num + f_num + cl_num + br_num + i_num - n_num) / 2
                                                            if 10 ** 6 * abs((c_num * exact_default.get('C') + h_num * exact_default.get('H') +o_num * exact_default.get(
                                                                        'O') + n_num * exact_default.get('N') + s_num * exact_default.get('S') + p_num * exact_default.get('P') +
                                                                              f_num * exact_default.get('F') + cl_num * exact_default.get(
                                                                        'Cl') + br_num * exact_default.get('Br') + i_num * exact_default.get('I')) - (float(mz) - 1.00728)) / float( mz) < ms1_error:
                                                                elements = np.array(['C', 'H', 'F', 'Cl', 'Br', 'I', 'N', 'O', 'P','S', ])
                                                                if element_limit[10] <= unsaturation <= element_limit[21]:
                                                                    if (h_num + f_num + cl_num + br_num + i_num - 2) / 2 < c_num and c_num < 2 * ( h_num + f_num + cl_num + br_num + i_num) + 2:
                                                                        ele_loc = np.where(elements_nums >= 1)
                                                                        formula = ''
                                                                        for l in ele_loc[0]:
                                                                            if elements_nums[l] == 1:
                                                                                formula = (str(formula) + str(elements[l]))
                                                                            else:
                                                                                formula = (str(formula) + str(elements[l]) + str(elements_nums[l]))
                                                                        formula_0.append(formula)

                    number = Counter(formula_0)
                    # 使用most_common()函数
                    result = number.most_common()
                    # 将结果打印出来
                    formula_final = ''
                    if len(result) >= 1:
                        for x in range(len(result)):
                            formula_final = formula_final + '; ' + result[x][0]
                    if len(result) == 0:
                        formula_final = ''
                    formula_final = formula_final.strip('; ')
                    ClassificationFormula.loc[i, 'Molecular formula'] = formula_final
                    percentage = int((i/len(ClassificationFormula_MS1)) * 100)
                    if percentage <= 99:
                        self.classification_signal.emit(json.dumps(percentage))
                    if percentage > 99:
                        self.classification_signal.emit(json.dumps(99))
        ClassificationFormula.replace('', np.nan, inplace=True)
        ClassificationFormula.dropna(subset=['Molecular formula'], axis=0, inplace=True)
        ClassificationFormula_path = 'results/AIClassification_FormulaFiltration_' + str(mgfpath.split('/')[-1].split('.')[-2]) + '.xlsx'
        ClassificationFormula.to_excel(ClassificationFormula_path,index=False)
        file_show_info = {'AIClassification_path': ClassificationFormula_path}
        self.show_signal.emit((json.dumps(file_show_info)))
        self.classification_signal.emit(json.dumps(100))

class MyMainForm(Ui_MainWindow, QMainWindow):

    def __init__(self):
        super().__init__()
        # self.setWindowFlags(Qt.WindowCloseButtonHint)  # 只显示关闭按钮
        self.setupUi(self)  # 初始化窗体设置
        qr = self.frameGeometry()  #显示至页面中心
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.setWindowFlags(Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)  # 显示关闭按钮和最小化按钮
        self.setWindowFlags(Qt.MSWindowsFixedSizeDialogHint)
        # 设置窗口的图标，引用当前目录下的web.png图片
        self.setWindowIcon(QIcon('supports/Icon.png'))
        self.mzxmlpath = False
        self.peaklistpath = False
        self.mgfpath=False
        self.eleConspath = 'supports/ElementComposition.xlsx'
        self.AIClassification=False
        self.AIClassification_path = False
        self.rt_max = False
        self.output_excel=False
        self.output_path = False
        self.index_no = 1
        self.progressBar.setValue(0)
        self.actionmzXML_file.triggered.connect(self.mzXMLFileOpen)  ##菜单栏的action打开文件
        self.actionPeakList_file.triggered.connect(self.PeakListFileOpen)
        self.actionMGF_file.triggered.connect(self.mgfFileOpen)
        self.pushButton.setCursor(QCursor(Qt.PointingHandCursor))
        self.pushButton_2.setCursor(QCursor(Qt.PointingHandCursor)) # 鼠标移动至按钮时，更改形状为手状
        self.pushButton_3.setCursor(QCursor(Qt.PointingHandCursor))
        self.pushButton_4.setCursor(QCursor(Qt.PointingHandCursor))
        self.pushButton_5.setCursor(QCursor(Qt.PointingHandCursor))

        self.pushButton.clicked.connect(self.Classification)  # 给pushButton_3添加一个点击事件
        self.pushButton_2.clicked.connect(self.FormulaCalculation)  # 给pushButton_3添加一个点击事件
        self.pushButton_3.clicked.connect(self.ShowLast)
        self.pushButton_4.clicked.connect(self.ShowNext)
        self.pushButton_5.clicked.connect(self.Save)


        self.tableWidget.setColumnCount(6)
        self.tableWidget.setColumnWidth(0, 40)
        self.tableWidget.setColumnWidth(1, 60)
        self.tableWidget.setColumnWidth(2, 50)
        self.tableWidget.setColumnWidth(3, 75)
        self.tableWidget.setColumnWidth(4, 55)
        self.tableWidget.setColumnWidth(5, 150)
        self.tableWidget.setHorizontalHeaderLabels([
            'No.', 'File', 'Scan', 'MS1', 'RT', 'Pred_class'])
        # 禁用双击编辑单元格
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 改为选择一行
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        # 添加右击菜单
        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.generate_menu)
        self.tableWidget_2.setColumnCount(2)
        self.tableWidget_2.setColumnWidth(0, 165)
        self.tableWidget_2.setColumnWidth(1, 165)
        self.tableWidget_2.setHorizontalHeaderLabels(['m/z', 'intensity'])
        # 禁用双击编辑单元格
        self.tableWidget_2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.Process = DrawFunction()  # process对象包含了所有的信号处理函数及其画图
        self.ImageLayout()

    def ImageLayout(self):
        self.fig1 = plt.figure()
        self.canvas1 = FigureCanvas(self.fig1)
        layout1 = QVBoxLayout()  # 垂直布局
        layout1.addWidget(self.canvas1)
        self.graphicsView.setLayout(layout1)  # 设置好布局之后调用函数

        self.fig2 = plt.figure()
        self.canvas2 = FigureCanvas(self.fig2)
        layout2 = QVBoxLayout()  # 垂直布局
        layout2.addWidget(self.canvas2)
        self.graphicsView_2.setLayout(layout2)  # 设置好布局之后调用函数

    def ClassificationStaus(self, staus):
        percentage = json.loads(staus)
        self.progressBar.setValue(percentage)
        if percentage == 100:
            self.textBrowser.append('******Analyse succeed!****** ')

    def InitialImage(self, file):
        file_show_info = json.loads(file)
        self.AIClassification_path = file_show_info.get('AIClassification_path')
        self.AIClassification=pd.read_excel(self.AIClassification_path)
        self.rt_max = file_show_info.get('rt_max')
        self.output_excel=pd.read_excel(self.AIClassification_path)
        self.output_excel['Peak recognition']=''
        # 1. 判断是否存储文件，如果不存在则函数结束
        if not os.path.exists(self.AIClassification_path):
            return
        # 2. 加载所有数据
        self.file_show = self.AIClassification['File']
        self.scan_show = self.AIClassification['Scan']
        self.ms1_show = self.AIClassification['Precursor ion']
        self.rt_show = self.AIClassification['RT']
        self.class_show = self.AIClassification['Predicted class']
        self.ms2_show = self.AIClassification['MSMS']
        self.probability_show = self.AIClassification['Probability']
        self.infos_count = len(self.file_show)
        self.tableWidget.setRowCount(self.infos_count)
        for i in range(self.infos_count):
            new_item1 = QTableWidgetItem(str(i + 1))
            self.tableWidget.setItem(i, 0, new_item1)
            new_item2 = QTableWidgetItem(str(self.file_show[i]))
            self.tableWidget.setItem(i, 1, new_item2)
            new_item3 = QTableWidgetItem(str(self.scan_show[i]))
            self.tableWidget.setItem(i, 2, new_item3)
            new_item4 = QTableWidgetItem(str(self.ms1_show[i]))
            self.tableWidget.setItem(i, 3, new_item4)
            new_item5 = QTableWidgetItem(str(self.rt_show[i]))
            self.tableWidget.setItem(i, 4, new_item5)
            new_item6 = QTableWidgetItem(str(self.class_show[i]))
            self.tableWidget.setItem(i, 5, new_item6)
        self.lineEdit_1.setText(str(1))
        self.lineEdit_2.setText(str(self.ms1_show[0]))
        self.lineEdit_3.setText(str(self.rt_show[0]))
        self.lineEdit_4.setText(str(self.class_show[0]))
        self.lineEdit_5.setText(str(round(self.probability_show[0], 2)))
        ms2_show_1 = self.ms2_show[0].split(';')
        infos_count1 = len(ms2_show_1)
        self.tableWidget_2.setRowCount(infos_count1)
        for l in range(infos_count1):
            new_item7 = QTableWidgetItem(str(ms2_show_1[l].split(' ')[0]))
            self.tableWidget_2.setItem(l, 0, new_item7)
            new_item8 = QTableWidgetItem(str(round(float(ms2_show_1[l].split(' ')[1]) * 100, 1)))
            self.tableWidget_2.setItem(l, 1, new_item8)
        self.label_3.setText(('MSMS Spectrum' + '        MS2@ ' + str(self.ms1_show[0]) + '   Scan: ' + str(
            self.scan_show[0]) + '   RT: ' + str(self.rt_show[0])))
        self.Process.Image_In_Process(self)  ##把实例传入进去
    def InitialImageII(self,file):
        file_show_info = json.loads(file)
        self.AIClassification_path = file_show_info.get('AIClassification_path')
        self.AIClassification = pd.read_excel(self.AIClassification_path)
        self.output_excel = pd.read_excel(self.AIClassification_path)
        self.output_excel['Peak recognition'] = ''
        # 1. 判断是否存储文件，如果不存在则函数结束
        if not os.path.exists(self.AIClassification_path):
            return
        # 2. 加载所有数据
        self.file_show = self.AIClassification['File']
        self.scan_show = self.AIClassification['Scan']
        self.ms1_show = self.AIClassification['Precursor ion']
        self.rt_show = self.AIClassification['RT']
        self.class_show = self.AIClassification['Predicted class']
        self.probability_show = self.AIClassification['Probability']
        self.infos_count = len(self.file_show)
        self.tableWidget.setRowCount(self.infos_count)
        for i in range(self.infos_count):
            new_item1 = QTableWidgetItem(str(i + 1))
            self.tableWidget.setItem(i, 0, new_item1)
            new_item2 = QTableWidgetItem(str(self.file_show[i]))
            self.tableWidget.setItem(i, 1, new_item2)
            new_item3 = QTableWidgetItem(str(self.scan_show[i]))
            self.tableWidget.setItem(i, 2, new_item3)
            new_item4 = QTableWidgetItem(str(self.ms1_show[i]))
            self.tableWidget.setItem(i, 3, new_item4)
            new_item5 = QTableWidgetItem(str(self.rt_show[i]))
            self.tableWidget.setItem(i, 4, new_item5)
            new_item6 = QTableWidgetItem(str(self.class_show[i]))
            self.tableWidget.setItem(i, 5, new_item6)
        self.lineEdit_1.setText('')
        self.lineEdit_2.setText('')
        self.lineEdit_3.setText('')
        self.lineEdit_4.setText('')
        self.lineEdit_5.setText('')

        self.tableWidget_2.setRowCount(0)
        self.label_3.setText('')
        self.Process.Image_In_ProcessII(self)  ##把实例传入进去

    def generate_menu(self, pos):
        """右键菜单"""
        if self.mzxmlpath:
            menu = QMenu()
            item = menu.addAction("Detailed information")
            action = menu.exec_(self.tableWidget.mapToGlobal(pos))
            if action == item:
                self.output_excel.loc[int(self.index_no)-1,'Peak recognition']=self.lineEdit_7.text()
                table_selected_index = self.tableWidget.currentIndex().row()
                model = self.tableWidget.model()
                self.index_no = model.data(model.index(table_selected_index, 0))
                self.lineEdit_1.setText(str(self.index_no))
                self.lineEdit_2.setText(str(self.ms1_show[(int(self.index_no)-1)]))
                self.lineEdit_3.setText(str(self.rt_show[(int(self.index_no)-1)]))
                self.lineEdit_4.setText(str(self.class_show[(int(self.index_no)-1)]))
                self.lineEdit_5.setText(str(round(self.probability_show[(int(self.index_no)-1)], 2)))
                self.lineEdit_7.setText('')
                ms2_show_1 = self.ms2_show[(int(self.index_no)-1)].split(';')
                infos_count1 = len(ms2_show_1)
                self.tableWidget_2.setRowCount(infos_count1)
                msms = []
                msms_inten = []
                for l in range(infos_count1):
                    msms.append(float((ms2_show_1[l].split(' ')[0])))
                    msms_inten.append((round(float(ms2_show_1[l].split(' ')[1]) * 100, 1)))
                    new_item7 = QTableWidgetItem(str(ms2_show_1[l].split(' ')[0]))
                    self.tableWidget_2.setItem(l, 0, new_item7)
                    new_item8 = QTableWidgetItem(str(round(float(ms2_show_1[l].split(' ')[1]) * 100, 1)))
                    self.tableWidget_2.setItem(l, 1, new_item8)
                self.label_3.setText(('MSMS Spectrum' + '        MS2@ ' + str(self.ms1_show[(int(self.index_no)-1)]) + '   Scan: ' + str(
                    self.scan_show[(int(self.index_no)-1)]) + '   RT: ' + str(self.rt_show[(int(self.index_no)-1)])))
                self.Process.Image_In_Process(self)  ##把实例传入进去

    def ShowLast(self):
        if self.mzxmlpath:
            if 0< (int(self.lineEdit_1.text()) - 1) <= self.infos_count:
                self.index_no = int(self.lineEdit_1.text()) - 1
                self.output_excel.loc[int(self.index_no), 'Peak recognition'] = self.lineEdit_7.text()
                self.lineEdit_1.setText(str(self.index_no))
                self.lineEdit_2.setText(str(self.ms1_show[(int(self.index_no) - 1)]))
                self.lineEdit_3.setText(str(self.rt_show[(int(self.index_no) - 1)]))
                self.lineEdit_4.setText(str(self.class_show[(int(self.index_no) - 1)]))
                self.lineEdit_5.setText(str(round(self.probability_show[(int(self.index_no) -1)], 2)))
                self.lineEdit_7.setText('')
                ms2_show_1 = self.ms2_show[(int(self.index_no) - 1)].split(';')
                infos_count1 = len(ms2_show_1)
                self.tableWidget_2.setRowCount(infos_count1)
                msms = []
                msms_inten = []
                for l in range(infos_count1):
                    msms.append(float((ms2_show_1[l].split(' ')[0])))
                    msms_inten.append((round(float(ms2_show_1[l].split(' ')[1]) * 100, 1)))
                    new_item7 = QTableWidgetItem(str(ms2_show_1[l].split(' ')[0]))
                    self.tableWidget_2.setItem(l, 0, new_item7)
                    new_item8 = QTableWidgetItem(str(round(float(ms2_show_1[l].split(' ')[1]) * 100, 1)))
                    self.tableWidget_2.setItem(l, 1, new_item8)
                self.label_3.setText(
                    ('MSMS Spectrum' + '        MS2@ ' + str(self.ms1_show[(int(self.index_no) - 1)]) + '   Scan: ' + str(
                        self.scan_show[(int(self.index_no) - 1)]) + '   RT: ' + str(self.rt_show[(int(self.index_no) - 1)])))
                self.Process.Image_In_Process(self)  ##把实例传入进去

    def ShowNext(self):
        if self.mzxmlpath:
            if 0 < (int(self.lineEdit_1.text()) + 1) <= self.infos_count:
                self.index_no = int(self.lineEdit_1.text()) + 1
                self.output_excel.loc[int(self.index_no) - 2, 'Peak recognition'] = self.lineEdit_7.text()
                self.lineEdit_1.setText(str(self.index_no))
                self.lineEdit_2.setText(str(self.ms1_show[(int(self.index_no) - 1)]))
                self.lineEdit_3.setText(str(self.rt_show[(int(self.index_no) - 1)]))
                self.lineEdit_4.setText(str(self.class_show[(int(self.index_no) - 1)]))
                self.lineEdit_5.setText(str(round(self.probability_show[(int(self.index_no) - 1)], 2)))
                self.lineEdit_7.setText('')
                ms2_show_1 = self.ms2_show[(int(self.index_no) - 1)].split(';')
                infos_count1 = len(ms2_show_1)
                self.tableWidget_2.setRowCount(infos_count1)
                msms = []
                msms_inten = []
                for l in range(infos_count1):
                    msms.append(float((ms2_show_1[l].split(' ')[0])))
                    msms_inten.append((round(float(ms2_show_1[l].split(' ')[1]) * 100, 1)))
                    new_item7 = QTableWidgetItem(str(ms2_show_1[l].split(' ')[0]))
                    self.tableWidget_2.setItem(l, 0, new_item7)
                    new_item8 = QTableWidgetItem(str(round(float(ms2_show_1[l].split(' ')[1]) * 100, 1)))
                    self.tableWidget_2.setItem(l, 1, new_item8)
                self.label_3.setText(
                    ('MSMS Spectrum' + '        MS2@ ' + str(self.ms1_show[(int(self.index_no) - 1)]) + '   Scan: ' + str(
                        self.scan_show[(int(self.index_no) - 1)]) + '   RT: ' + str(
                        self.rt_show[(int(self.index_no) - 1)])))
                self.Process.Image_In_Process(self)  ##把实例传入进去

    def Save(self):
        self.output_excel.loc[int(self.index_no) - 1, 'Peak recognition'] = self.lineEdit_7.text()
        self.output_path=str(self.AIClassification_path).split('.xlsx')[0]+'_peak_recognition'+'.xlsx'
        self.output_excel.to_excel(self.output_path,index=False)
        self.textBrowser.append('*****Save!******')


    def mzXMLFileOpen(self):  ##打开文件
        self.mzxmlpath, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'MS Data (*.mzXML)')
        if self.mzxmlpath:  ##选中文件之后就选中了需要播放的音乐，并同时显示出来
            self.textBrowser.append('*****Open mzXML file : ' + str(self.mzxmlpath.split('/')[-1]) + '******')
            self.mgfpath=False
        if self.mzxmlpath=='':
            self.mzxmlpath=False

    def PeakListFileOpen(self):  ##打开文件
        self.peaklistpath, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'Peak List (*.xlsx)')
        if self.peaklistpath:  ##选中文件之后就选中了需要播放的音乐，并同时显示出来
            self.textBrowser.append('*****Open peak list file : ' + str(self.peaklistpath.split('/')[-1]) + '******')
        if self.peaklistpath=='':
            self.peaklistpath=False

    def mgfFileOpen(self):
        self.mgfpath, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'MS Data (*.mgf)')
        if self.mgfpath:  ##选中文件之后就选中了需要播放的音乐，并同时显示出来
            self.textBrowser.append('*****Open mgf file : ' + str(self.mgfpath.split('/')[-1]) + '******')
            self.mzxmlpath=False
        if self.mgfpath=='':
            self.mgfpath=False

    def Classification(self):  ##这里对应的是打开文件，并点击按钮
        if self.mzxmlpath and self.peaklistpath and self.mgfpath==False:
            input_path = json.dumps({'mzxmlpath': self.mzxmlpath, 'peaklistpath': self.peaklistpath})
            self.classification_thread1 = MyThreadOne()
            self.classification_thread1.classification_signal.connect(self.ClassificationStaus)
            self.classification_thread1.show_signal.connect(self.InitialImage)
            self.classification_thread1.input_path = input_path
            self.classification_thread1.start()
            self.textBrowser.append('**AIClassification in progress : '+str(self.mzxmlpath.split('/')[-1])+'**'+'peakList : '+str(self.peaklistpath.split('/')[-1]))

        if self.mzxmlpath and self.peaklistpath==False and self.mgfpath==False:
            reply = QMessageBox.question(self, 'Please Note',
                                         'Are you sure there is no peak list file?',
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                input_path = json.dumps({'mzxmlpath': self.mzxmlpath})
                self.classification_thread11 = MyThreadOneII()
                self.classification_thread11.classification_signal.connect(self.ClassificationStaus)
                self.classification_thread11.show_signal.connect(self.InitialImage)
                self.classification_thread11.input_path = input_path
                self.classification_thread11.start()
                self.textBrowser.append('******AIClassification in progress : '+str(self.mzxmlpath.split('/')[-1])+'******')
        if self.mzxmlpath==False and self.mgfpath and self.peaklistpath:
            input_path = json.dumps({'mgfpath': self.mgfpath, 'peaklistpath': self.peaklistpath})
            self.classification_thread2 = MyThreadTwo()
            self.classification_thread2.classification_signal.connect(self.ClassificationStaus)
            self.classification_thread2.show_signal.connect(self.InitialImageII)
            self.classification_thread2.input_path = input_path
            self.classification_thread2.start()
            self.textBrowser.append('**AIClassification in progress : '+str(self.mgfpath.split('/')[-1])+'**'+'peakList : '+str(self.peaklistpath.split('/')[-1]))
        if self.mgfpath and self.peaklistpath==False and self.mzxmlpath==False:
            reply = QMessageBox.question(self, 'Please Note',
                                         'Are you sure there is no peak list file?',
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                input_path = json.dumps({'mgfpath': self.mgfpath})
                self.classification_thread21 = MyThreadTwoII()
                self.classification_thread21.classification_signal.connect(self.ClassificationStaus)
                self.classification_thread21.show_signal.connect(self.InitialImageII)
                self.classification_thread21.input_path = input_path
                self.classification_thread21.start()
                self.textBrowser.append('******AIClassification in progress : '+str(self.mgfpath.split('/')[-1])+'******')

        if self.mzxmlpath==False and self.mgfpath==False:
            reply = QMessageBox.question(self, 'Warning',
                                         'Please open a MS data (mzXML or mgf file)',
                                         QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                pass

    def FormulaCalculation(self):
        self.progressBar.setValue(0)
        if self.mzxmlpath and self.AIClassification_path and self.mgfpath==False:
            input_path = json.dumps({'eleConspath':self.eleConspath,'AIClassification_path': self.AIClassification_path,'mzxmlpath': self.mzxmlpath,'rt_max':self.rt_max})
            self.classification_thread3 = MyThreadThree()
            self.classification_thread3.classification_signal.connect(self.ClassificationStaus)
            self.classification_thread3.show_signal.connect(self.InitialImage)
            self.classification_thread3.input_path = input_path
            self.classification_thread3.start()
            self.textBrowser.append('******Formula filtration in progress : '+str(self.mzxmlpath.split('/')[-1])+'******')
        if self.mgfpath and self.AIClassification_path and self.mzxmlpath==False:
            input_path = json.dumps({'eleConspath':self.eleConspath,'AIClassification_path': self.AIClassification_path,'mgfpath': self.mgfpath})
            self.classification_thread4 = MyThreadFour()
            self.classification_thread4.classification_signal.connect(self.ClassificationStaus)
            self.classification_thread4.show_signal.connect(self.InitialImageII)
            self.classification_thread4.input_path = input_path
            self.classification_thread4.start()
            self.textBrowser.append('******Formula filtration in progress : '+str(self.mgfpath.split('/')[-1])+'******')
        if self.AIClassification_path == False:
            QMessageBox.information(self, 'Please Note', 'Please complete the class prediction in the first step')
        if self.AIClassification_path and self.mzxmlpath == False and self.mgfpath == False:
            QMessageBox.information(self, 'Please Note', 'Please complete the class prediction in the first step')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainForm()
    myWin.show()
    sys.exit(app.exec_())