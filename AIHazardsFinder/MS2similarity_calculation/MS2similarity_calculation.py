import pandas as pd
import os
import numpy as np
from pyteomics import mgf
import spectral_entropy
import tools

data_df=pd.read_csv(r"MoNA-export-MassBank-sdf\MoNA-export-MassBank.csv")

precursorMass = data_df['EXACT_MASS']
precursorMass = np.array(precursorMass).tolist()
num = data_df.shape[0]
cid=data_df['NAME']
cid = np.array(cid).tolist()

smiles=data_df['SMILES']
smiles = np.array(smiles).tolist()

cas=data_df['INCHI']
cas = np.array(cas).tolist()

MSMS_database=data_df['SPECTRAL']


output=pd.DataFrame()


dir=r"E:\BaiduSyncdisk\validation.mgf"

row=-1
file = mgf.read(dir)
for l in range(len(file)):
    print(l)
    mgf_rt = round(float(file[l]['params']['rtinseconds'] / 60), 2)
    precursor = round(float(file[l]['params']['pepmass'][0]), 5)
    file_doc = file[l]['params']['title'].split(':')[1].split(',')[0]
    spectral = file[l]['m/z array']
    spectral_inten = file[l]['intensity array']
    for i in range(num):

        if pd.isnull(MSMS_database[i]) == False:
            precursorMass_eve = precursorMass[i]+1.0073
            if 10**6*abs(precursor - precursorMass_eve) / precursorMass_eve < 5:
                spec_query = []
                for m in range(len(spectral)):
                    spec0 = []
                    spec0.append(spectral[m])
                    spec0.append(spectral_inten[m])
                    spec_query.append(spec0)
                spec_query = np.array(spec_query, dtype=np.float32)
                MSMS_reference = MSMS_database[i].split(';')
                spec_reference = []
                for n in range(len(MSMS_reference)):
                    spec0 = []
                    spec0.append(MSMS_reference[n].split(' ')[0])
                    spec0.append(MSMS_reference[n].split(' ')[1])
                    spec_reference.append(spec0)
                spec_reference = np.array(spec_reference, dtype=np.float32)
                spec_query_clean = tools.clean_spectrum(spec_query, max_mz=precursor, noise_removal=0.01,
                                                        ms2_ppm=10)
                spec_reference_clean = tools.clean_spectrum(spec_reference, max_mz=precursorMass_eve,
                                                            noise_removal=0.01, ms2_ppm=10)
                similarity = spectral_entropy.calculate_entropy_similarity(spec_query_clean, spec_reference_clean,
                                                                           ms2_da=0.01)
                if similarity>0.5:
                    cid_eve = cid[i]
                    row = row + 1
                    output.loc[row, 'sample'] = file_doc
                    output.loc[row, 'PubchemCID'] = cid_eve
                    output.loc[row, 'cas'] = cas[i]
                    output.loc[row, 'smiles'] = smiles[i]
                    output.loc[row, 'rt'] = mgf_rt
                    output.loc[row, 'ms2'] = precursor
                    output.loc[row, 'similarity'] = similarity



output.to_excel(r'E:\BaiduSyncdisk\validation_indentify.xlsx', index=False)




