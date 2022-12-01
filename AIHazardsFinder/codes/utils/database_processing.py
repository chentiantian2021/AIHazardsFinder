# -*- coding: utf-8 -*-
# Created by: ctt@2022
# chentiantian@dicp.ac.cn

import pandas as pd
import numpy as np
import math
import copy

def getListMaxNumIndex(num_list,topk=30):
    tmp_list=copy.deepcopy(num_list)
    tmp_list.sort()
    max_num_index=[num_list.index(one) for one in tmp_list[::-1][:topk]]
    return max_num_index


def removeNeu(spectral_neu, spectral_neu_inten, ms2_error,starNeu, endNeu):
    spectral_neu=np.array(spectral_neu)
    spectral_neu_inten=np.array(spectral_neu_inten)

    keep = np.where(spectral_neu > starNeu)
    spectral_neu = spectral_neu[keep]
    spectral_neu_inten = spectral_neu_inten[keep]
    keep = np.where(spectral_neu < endNeu)
    spectral_neu = spectral_neu[keep]
    spectral_neu_inten = spectral_neu_inten[keep]
    for l in range(len(spectral_neu)):
        for m in range(l, len(spectral_neu)):
            if l != m and abs(spectral_neu[l] - spectral_neu[m]) < ms2_error:
                if spectral_neu_inten[l] > spectral_neu_inten[m]:
                    spectral_neu_inten[m] = 0
                    spectral_neu_inten[m] = 0
                else:
                    spectral_neu_inten[l] = 0
                    spectral_neu_inten[l] = 0
    spectral_neu = list(spectral_neu)
    spectral_neu_inten = list(spectral_neu_inten)
    max_num_index = getListMaxNumIndex(spectral_neu_inten, topk=30)
    spectral_neu = np.array(spectral_neu)[max_num_index]
    spectral_neu_inten = np.array(spectral_neu_inten)[max_num_index]
    return spectral_neu, spectral_neu_inten


def vector_binning(database , endMass , starMass , starNeu , endNeu , bin_width , ms2_error ,precursormz ,ms2, categories):
    bin_num = (endMass - starMass) / bin_width
    bin_neu_num = (endNeu - starNeu) / bin_width
    database_df1 = pd.DataFrame(0, index=list(range(int(database.shape[0]))), columns=list(range(round(bin_num))))
    database_df2 = pd.DataFrame(0, index=list(range(int(database.shape[0]))), columns=list(range(round(bin_neu_num))))
    remove_list = []
    for i in range(database.shape[0]):
        print(i)
        ms2_split = ms2[i].split(';')
        spectral = []
        spectral_inten = []
        for l in range(len(ms2_split)):
            spectral.append(float(ms2_split[l].split(' ')[0]))
            spectral_inten.append(float(ms2_split[l].split(' ')[1]))
        spectral = np.array(spectral)
        spectral_inten = np.array(spectral_inten)
        spectral_inten = spectral_inten / max(spectral_inten)
        keep = np.where(spectral_inten >0.01)
        spectral = spectral[keep]
        spectral_inten = spectral_inten[keep]

        keep = np.where(spectral - precursormz[i] < 0.01)
        spectral = spectral[keep]
        spectral_inten = spectral_inten[keep]

        keep=np.where(spectral < endMass)
        spectral=spectral[keep]
        spectral_inten=spectral_inten[keep]

        keep = np.where(spectral > starMass)
        spectral = spectral[keep]
        spectral_inten = spectral_inten[keep]

        spectral_inten=np.sqrt(spectral_inten)

        if len(spectral) <5:
            remove_list.append(i)
        for l in range(len(spectral)):
            bin_position = math.floor((float(spectral[l]) - starMass) / bin_width)
            database_df1.iloc[i,bin_position] = max(round(float(spectral_inten[l]), 3),database_df1.iloc[i,bin_position])
        spectral_neu=[]
        spectral_neu_inten=[]
        for n in range(len(spectral)):
            spectral_neu.append(abs(precursormz[i]-spectral[n]))
            spectral_neu_inten.append((spectral_inten[n]))
        spectral_neu, spectral_neu_inten = removeNeu(spectral_neu, spectral_neu_inten, ms2_error,starNeu, endNeu)
        for l in range(len(spectral_neu)):
            bin_neu_position = math.floor((float(spectral_neu[l]) - starNeu) / bin_width)
            database_df2.iloc[i, bin_neu_position] = max(round(float(spectral_neu_inten[l]), 3),database_df2.iloc[i, bin_neu_position])
    X = np.concatenate((database_df1, database_df2), axis=1)
    X = np.delete(X, remove_list, 0)

    Y=pd.Categorical(categories).codes
    Y = np.delete(Y, remove_list, 0)
    classes = list(set(Y))
    return X, Y, classes

def vector_binning_crossvalidationII(database , endMass , starMass , starNeu , endNeu , bin_width , ms2_error ,precursormz ,ms2, categories,inchikeies):
    bin_num = (endMass - starMass) / bin_width
    bin_neu_num = (endNeu - starNeu) / bin_width
    database_df1 = pd.DataFrame(0, index=list(range(int(database.shape[0]))), columns=list(range(round(bin_num))))
    database_df2 = pd.DataFrame(0, index=list(range(int(database.shape[0]))), columns=list(range(round(bin_neu_num))))
    inchikey_list=[]
    remove_list = []
    for i in range(database.shape[0]):
        print(i)
        inchikey_list.append(inchikeies[i])
        ms2_split = ms2[i].split(';')
        spectral = []
        spectral_inten = []
        for l in range(len(ms2_split)):
            spectral.append(float(ms2_split[l].split(' ')[0]))
            spectral_inten.append(float(ms2_split[l].split(' ')[1]))
        spectral = np.array(spectral)
        spectral_inten = np.array(spectral_inten)
        spectral_inten = spectral_inten / max(spectral_inten)
        keep = np.where(spectral_inten >0.01)
        spectral = spectral[keep]
        spectral_inten = spectral_inten[keep]

        keep = np.where(spectral - precursormz[i] < 0.01)
        spectral = spectral[keep]
        spectral_inten = spectral_inten[keep]

        keep=np.where(spectral < endMass)
        spectral=spectral[keep]
        spectral_inten=spectral_inten[keep]

        keep = np.where(spectral > starMass)
        spectral = spectral[keep]
        spectral_inten = spectral_inten[keep]

        spectral_inten=np.sqrt(spectral_inten)
        if len(spectral) <5:
            remove_list.append(i)
        for l in range(len(spectral)):
            bin_position = math.floor((float(spectral[l]) - starMass) / bin_width)
            database_df1.iloc[i,bin_position] = max(round(float(spectral_inten[l]), 3),database_df1.iloc[i,bin_position])
        spectral_neu=[]
        spectral_neu_inten=[]
        for n in range(len(spectral)):
            spectral_neu.append(abs(precursormz[i]-spectral[n]))
            spectral_neu_inten.append((spectral_inten[n]))
        spectral_neu, spectral_neu_inten = removeNeu(spectral_neu, spectral_neu_inten, ms2_error,starNeu, endNeu)

        for l in range(len(spectral_neu)):
            bin_neu_position = math.floor((float(spectral_neu[l]) - starNeu) / bin_width)
            database_df2.iloc[i, bin_neu_position] = max(round(float(spectral_neu_inten[l]), 3),database_df2.iloc[i, bin_neu_position])
    X= np.concatenate((database_df1,database_df2),axis=1)
    X=np.delete(X,remove_list,0)
    inchikey_list = [inchikey_list[g] for g in range(len(inchikey_list)) if (g not in remove_list)]

    Y=pd.Categorical(categories).codes
    Y = np.delete(Y, remove_list, 0)
    classes = list(set(Y))
    inchikey_list_uni = []
    g=-1
    index=[]
    for li in inchikey_list:
        g=g+1
        if li not in inchikey_list_uni:
            index.append(g)
            inchikey_list_uni.append(li)
    X_uni = [X[i] for i in range(len(X)) if (i in index)]
    Y_uni = [Y[i] for i in range(len(Y)) if (i in index)]
    return X, Y, classes, inchikey_list, inchikey_list_uni, X_uni, Y_uni

def data_load_crossvalidationII(X, Y, inchikey_list, inchikey_list_uni,train_index, test_index):
    train_inchikey = [inchikey_list_uni[i] for i in range(len(inchikey_list_uni)) if (i in train_index)]
    test_inchikey = [inchikey_list_uni[i] for i in range(len(inchikey_list_uni)) if (i in test_index)]
    new_train_index=[]
    new_test_index=[]
    for l in range(len(train_inchikey)):
        for g in range(len(inchikey_list)):
            if train_inchikey[l]==inchikey_list[g]:
                new_train_index.append(g)
    for l in range(len(test_inchikey)):
        for g in range(len(inchikey_list)):
            if test_inchikey[l]==inchikey_list[g]:
                new_test_index.append(g)
    new_X_train = X[new_train_index]
    new_X_test = X[new_test_index]
    new_Y_train = Y[new_train_index]
    new_Y_test = Y[new_test_index]
    return new_X_train, new_X_test, new_Y_train, new_Y_test
