# -*- coding: utf-8 -*-
# Created by: ctt@2022
# chentiantian@dicp.ac.cn

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report
from utils import database_processing
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier
from sklearn.svm import SVC

starMass = 50
endMass = 800
bin_width = 0.01
starNeu = 12
endNeu = 200
ms2_error=0.01

database=pd.read_excel(r"E:\第二个工作文章所用数据\方法学考察\直接分类的全部数据-最终版本-包括其他类别-22-7-13.xlsx")
precursormz = database['MS1']
ms2 = database['MS2']
categories=database['Class label']
inchikeies=database['Inchikey']

def mlp_test(model,X_test,Y_test):
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    y_true = Y_test
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    result_list1 = [precision_micro, precision_macro, recall_micro, recall_macro, f1_micro, f1_macro]
    class_eve_classification = classification_report(y_true, y_pred,output_dict=True)
    result_list2=[]
    for cls in classes:
        every_class = []
        every_class.append(class_eve_classification[str(cls)]['precision'])
        every_class.append(class_eve_classification[str(cls)]['recall'])
        every_class.append(class_eve_classification[str(cls)]['f1-score'])
        result_list2.append(every_class)
    return result_list1, result_list2

def rf_svm_test(model,X_test,Y_test):
    y_pred = model.predict(X_test)
    y_true = Y_test
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    result_list1 = [precision_micro, precision_macro, recall_micro, recall_macro, f1_micro, f1_macro]
    class_eve_classification = classification_report(y_true, y_pred,output_dict=True)
    result_list2=[]
    for cls in classes:
        every_class = []
        every_class.append(class_eve_classification[str(cls)]['precision'])
        every_class.append(class_eve_classification[str(cls)]['recall'])
        every_class.append(class_eve_classification[str(cls)]['f1-score'])
        result_list2.append(every_class)
    return result_list1, result_list2

def mlp_crossvali(X_train,Y_train,class_weights,X_test,Y_test):
    print('mlp正在训练。。。')
    mlp_clf = tf.keras.Sequential()
    mlp_clf.add(Dense(128, input_shape=(93800,), activation='relu'))
    mlp_clf.add(Dense(33, activation='softmax'))
    mlp_clf.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    mlp_clf.fit(X_train,Y_train, batch_size=64, epochs=10, class_weight=class_weights)
    result_list1, result_list2 = mlp_test(mlp_clf, X_test, Y_test)
    return result_list1, result_list2

def rf_crossvali(X_train,Y_train,class_weights,X_test,Y_test):
    print('rf正在训练。。。')
    rf_clf = RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=1, class_weight= class_weights, n_jobs=14, verbose=1)
    rf_clf.fit(X_train, Y_train)
    # max_depth=3,
    result_list1, result_list2=rf_svm_test(rf_clf,X_test,Y_test)
    return result_list1, result_list2

def xgboost_crossvali(X_train,Y_train,class_weights,X_test,Y_test):
    print('xgboost正在训练。。。')
    xgb_clf = XGBClassifier(booster='gbtree', objective='multi:softmax',eval_metric='mlogloss', num_class=33,
                            n_estimators=50,
                            random_state=1,n_jobs=14)
    xgb_clf.fit(X_train,Y_train)
    result_list1, result_list2 = rf_svm_test(xgb_clf, X_test, Y_test)
    return result_list1, result_list2

def svm_crossvali(X_train,Y_train,class_weights,X_test,Y_test):
    print('svm正在训练。。。')
    svm_clf = SVC(kernel='rbf', C=1, class_weight=class_weights, random_state=1,verbose=1)
    #gamma = 0.1
    svm_clf.fit(X_train,Y_train)
    result_list1, result_list2 ,= rf_svm_test(svm_clf, X_test, Y_test)
    return result_list1, result_list2

# crossvalidation II
X, Y, classes, inchikey_list, inchikey_list_uni, X_uni, Y_uni = database_processing.vector_binning_crossvalidationII(
    database , endMass , starMass , starNeu , endNeu , bin_width , ms2_error ,precursormz ,ms2, categories,inchikeies)

Y=Y-1
classes=classes-1
Y_uni=Y_uni-1
mlp_result1=pd.DataFrame()
mlp_result2=pd.DataFrame()
rf_result1=pd.DataFrame()
rf_result2=pd.DataFrame()
xgboost_result1=pd.DataFrame()
xgboost_result2=pd.DataFrame()
svm_result1=pd.DataFrame()
svm_result2=pd.DataFrame()
i = -1
rangdom_list=list(range(10))
for seq in range(len(rangdom_list)):
    print('第几个循环  ',seq)
    skf = StratifiedKFold(n_splits=10, random_state=rangdom_list[seq], shuffle=True)
    for train_index, test_index in skf.split(X_uni,Y_uni):
        i=i+1
        new_X_train, new_X_test, new_Y_train, new_Y_test=database_processing.data_load_crossvalidationII(X, Y, inchikey_list, inchikey_list_uni,train_index, test_index)
        class_weights = dict()
        for l in range(len(classes)):
            num = 0
            for g in range(len(new_Y_train)):
                if l == new_Y_train[g]:
                    num = num + 1
            class_weights[l] = len(new_Y_train) / (num * 33)
        result_list1, result_list2 = mlp_crossvali(new_X_train, new_Y_train, class_weights, new_X_test, new_Y_test)
        mlp_result1.loc[i, 'precision_micro'] = result_list1[0]
        mlp_result1.loc[i, 'precision_macro'] = result_list1[1]
        mlp_result1.loc[i, 'recall_micro'] = result_list1[2]
        mlp_result1.loc[i, 'recall_macro'] = result_list1[3]
        mlp_result1.loc[i, 'f1_micro'] = result_list1[4]
        mlp_result1.loc[i, 'f1_macro'] = result_list1[5]
        for l in range(len(classes)):
            mlp_result2.loc[i, str(classes[l]) + '_precision'] = result_list2[l][0]
            mlp_result2.loc[i, str(classes[l]) + '_recall'] = result_list2[l][1]
            mlp_result2.loc[i, str(classes[l]) + '_f1-score'] = result_list2[l][2]
        result_list1, result_list2=rf_crossvali(new_X_train,new_Y_train,class_weights,new_X_test,new_Y_test)
        rf_result1.loc[i,'precision_micro']=result_list1[0]
        rf_result1.loc[i, 'precision_macro'] = result_list1[1]
        rf_result1.loc[i, 'recall_micro'] = result_list1[2]
        rf_result1.loc[i, 'recall_macro'] = result_list1[3]
        rf_result1.loc[i, 'f1_micro'] = result_list1[4]
        rf_result1.loc[i, 'f1_macro'] = result_list1[5]
        for l in range(len(classes)):
            rf_result2.loc[i, str(classes[l]) + '_precision'] = result_list2[l][0]
            rf_result2.loc[i, str(classes[l]) + '_recall'] = result_list2[l][1]
            rf_result2.loc[i, str(classes[l]) + '_f1-score'] = result_list2[l][2]

        result_list1, result_list2= xgboost_crossvali(new_X_train, new_Y_train, class_weights, new_X_test, new_Y_test)
        xgboost_result1.loc[i, 'precision_micro'] = result_list1[0]
        xgboost_result1.loc[i, 'precision_macro'] = result_list1[1]
        xgboost_result1.loc[i, 'recall_micro'] = result_list1[2]
        xgboost_result1.loc[i, 'recall_macro'] = result_list1[3]
        xgboost_result1.loc[i, 'f1_micro'] = result_list1[4]
        xgboost_result1.loc[i, 'f1_macro'] = result_list1[5]
        for l in range(len(classes)):
            xgboost_result2.loc[i, str(classes[l]) + '_precision'] = result_list2[l][0]
            xgboost_result2.loc[i, str(classes[l]) + '_recall'] = result_list2[l][1]
            xgboost_result2.loc[i, str(classes[l]) + '_f1-score'] = result_list2[l][2]

        result_list1, result_list2= svm_crossvali(new_X_train, new_Y_train, class_weights, new_X_test, new_Y_test)
        svm_result1.loc[i, 'precision_micro'] = result_list1[0]
        svm_result1.loc[i, 'precision_macro'] = result_list1[1]
        svm_result1.loc[i, 'recall_micro'] = result_list1[2]
        svm_result1.loc[i, 'recall_macro'] = result_list1[3]
        svm_result1.loc[i, 'f1_micro'] = result_list1[4]
        svm_result1.loc[i, 'f1_macro'] = result_list1[5]
        for l in range(len(classes)):
            svm_result2.loc[i, str(classes[l]) + '_precision'] = result_list2[l][0]
            svm_result2.loc[i, str(classes[l]) + '_recall'] = result_list2[l][1]
            svm_result2.loc[i, str(classes[l]) + '_f1-score'] = result_list2[l][2]


for i in range(6):
    mlp_result1.loc[102, i] = mlp_result1.mean(axis=0)[i]
    rf_result1.loc[102,i]=rf_result1.mean(axis=0)[i]
    xgboost_result1.loc[102, i] = xgboost_result1.mean(axis=0)[i]
    svm_result1.loc[102, i] = svm_result1.mean(axis=0)[i]

for i in range(len(classes)):
    mlp_result2.loc[102, i] = mlp_result2.mean(axis=0)[i]
    rf_result2.loc[102, i] = rf_result2.mean(axis=0)[i]
    xgboost_result2.loc[102, i] = xgboost_result2.mean(axis=0)[i]
    svm_result2.loc[102, i] = svm_result2.mean(axis=0)[i]

mlp_result1.to_excel('各个模型交叉验证/MLP/all_crossvali.xlsx',index=False)
rf_result1.to_excel('各个模型交叉验证/RF/all_crossvali.xlsx',index=False)
xgboost_result1.to_excel('各个模型交叉验证/XGBOOST/all_crossvali.xlsx',index=False)
svm_result1.to_excel('各个模型交叉验证/SVM/all_crossvali.xlsx',index=False)

mlp_result2.to_excel('各个模型交叉验证/MLP/classes_crossvali.xlsx',index=False)
rf_result2.to_excel('各个模型交叉验证/RF/classes_crossvali.xlsx',index=False)
xgboost_result2.to_excel('各个模型交叉验证/XGBOOST/classes_crossvali.xlsx',index=False)
svm_result2.to_excel('各个模型交叉验证/SVM/classes_crossvali.xlsx',index=False)

