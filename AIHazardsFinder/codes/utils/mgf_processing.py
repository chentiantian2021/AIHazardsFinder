# -*- coding: utf-8 -*-
# Created by: ctt@2022
# chentiantian@dicp.ac.cn

import numpy as np
import copy


def getListMaxNumIndex(num_list,topk=30):
    tmp_list=copy.deepcopy(num_list)
    tmp_list.sort()
    max_num_index=[num_list.index(one) for one in tmp_list[::-1][:topk]]
    return max_num_index


def removeNeu(spectral_neu, spectral_neu_inten, ms2_error,starNeu, endNeu):
    for l in range(len(spectral_neu)):
        for m in range(l, len(spectral_neu)):
            if l != m and abs(spectral_neu[l] - spectral_neu[m]) < ms2_error:
                if spectral_neu_inten[l] > spectral_neu_inten[m]:
                    spectral_neu_inten[m] = 0
                    spectral_neu_inten[m] = 0
                else:
                    spectral_neu_inten[l] = 0
                    spectral_neu_inten[l] = 0
    max_num_index = getListMaxNumIndex(spectral_neu_inten, topk=30)
    spectral_neu = np.array(spectral_neu)[max_num_index]
    spectral_neu_inten = np.array(spectral_neu_inten)[max_num_index]
    keep = np.where(spectral_neu > starNeu)
    spectral_neu = spectral_neu[keep]
    spectral_neu_inten = spectral_neu_inten[keep]

    keep = np.where(spectral_neu < endNeu)
    spectral_neu = spectral_neu[keep]
    spectral_neu_inten = spectral_neu_inten[keep]
    return spectral_neu, spectral_neu_inten

def screening_vector_binning(spectral, spectral_inten, endMass , starMass , starNeu , endNeu , ms2_error ,precursormz):
    # spectral, spectral_inten = profile_to_centroid.centroid(spectral, spectral_inten)
    spectral = np.array(spectral)
    spectral_inten = np.array(spectral_inten)
    spectral_inten = spectral_inten / max(spectral_inten)
    keep = np.where(spectral_inten > 0.01)
    spectral = spectral[keep]
    spectral_inten = spectral_inten[keep]


    keep = np.where(spectral - precursormz < 0.01)
    spectral = spectral[keep]
    spectral_inten = spectral_inten[keep]

    keep=np.where(spectral < endMass)
    spectral=spectral[keep]
    spectral_inten=spectral_inten[keep]

    keep = np.where(spectral > starMass)
    spectral = spectral[keep]
    spectral_inten = spectral_inten[keep]

    spectral_inten=np.sqrt(spectral_inten)

    spectral_neu=[]
    spectral_neu_inten=[]

    for n in range(len(spectral)):
        spectral_neu.append(abs(precursormz-spectral[n]))
        spectral_neu_inten.append((spectral_inten[n]))
    spectral_neu, spectral_neu_inten = removeNeu(spectral_neu, spectral_neu_inten, ms2_error,starNeu, endNeu)
    return spectral, spectral_inten, spectral_neu,spectral_neu_inten
