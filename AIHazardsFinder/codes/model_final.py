# -*- coding: utf-8 -*-
# Created by: ctt@2022
# chentiantian@dicp.ac.cn

import pandas as pd
import numpy as np
from utils import database_processing
from tensorflow.keras.layers import Dense
from tensorflow import keras
from sklearn.utils import shuffle
starMass = 50
endMass = 800
bin_width = 0.02
starNeu = 12
endNeu = 200
ms2_error=0.01

database=pd.read_excel(r"data\直接分类的全部数据-最终版本-包括其他类别-22-7-10.xlsx")
precursormz = database['MS1']
ms2 = database['MS2']
categories=database['Class label']

X, Y, classes= database_processing.vector_binning(database , endMass , starMass , starNeu , endNeu , bin_width , ms2_error ,precursormz ,ms2, categories)
X,Y = shuffle(X,Y, random_state=66)

X,Y = shuffle(X,Y, random_state=18)
def model_load():
    model=keras.Sequential()
    model.add(Dense(128, input_shape=(93800,), activation='relu'))
    model.add(Dense(33,activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=[keras.metrics.sparse_categorical_accuracy])
    return model

def train(epochs):
    model = model_load()
    class_weight = dict()
    for cl in range(len(classes)):
        num = 0
        for g in range(len(Y)):
            if cl == Y[g]:
                num = num + 1
        class_weight[cl] = len(Y) / (num * 33)
    model.fit(X, Y, batch_size=64, epochs=epochs,class_weight=class_weight,shuffle=False)
    model.save('选择合适的模型20221023/classification_model_18.h5')

if __name__ == '__main__':
    train(epochs=10)  # todo epoch可以设置为你想要的轮数