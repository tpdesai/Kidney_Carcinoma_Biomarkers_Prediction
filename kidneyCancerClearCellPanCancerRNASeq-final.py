#!/usr/bin/env python
# coding: utf-8

# In[196]:


import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from random import sample
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import math


# ### Pre-processing of data and Normalizing the columns for Machine Learning

# In[197]:


# read in the data

data = pd.read_csv('clearCellPanCancerRNASeq.txt', sep = '\t')


# In[198]:


data


# In[199]:


data = data.drop(['Entrez_Gene_Id'], axis=1)
data


# In[200]:


# remove genes which were NaN

data = data.dropna(subset=['Hugo_Symbol'])
data


# In[201]:


# transpose the data so that genes are columns and rows are samples

data = np.transpose(data)
data


# In[202]:


# rename columns for dataframe

columns = data.iloc[0]


# In[203]:


data = data[1:]


# In[204]:


data.columns = columns
data


# ### Matching sample ID with metastasis status

# In[205]:


metastasis = pd.read_csv('clearCellPanCancerClinical.txt', sep = '\t')


# In[206]:


metastasis


# In[207]:


# obtain a list of sample ID

idList = metastasis['Sample ID'].tolist()
print(len(idList))


# In[208]:


metStatus = metastasis['American Joint Committee on Cancer Metastasis Stage Code'].tolist()


# In[209]:


dataID = data.index.tolist()


# In[210]:


dataID


# In[211]:


idList


# In[212]:


# create column for metastasis status in data dataframe
listDataStatus = []
for name in dataID:
    if name in idList:
        index = idList.index(name)
        listDataStatus.append(metStatus[index])
        


# In[213]:


listDataStatus


# In[214]:


data['status'] = listDataStatus


# In[215]:


data


# In[216]:


# filter 


# In[217]:


statusData = data[data['status'].isin(['M0','M1'])]


# In[218]:


statusData


# In[219]:


statusData = statusData.dropna(axis=1)


# In[220]:


m0 = statusData[statusData['status'] == 'M0']
m1 = statusData[statusData['status'] == 'M1']


# In[221]:



m0 = m0.drop('status', axis = 1)


# In[222]:


m1 = m1.drop('status', axis = 1)


# In[223]:


m0


# In[224]:


m1


# In[225]:


print(len(m0.columns))


# In[226]:


print(len(m1.columns))


# In[227]:


# make both dataframes have the same number of rows

newM1 = m1.sample(401, replace = True)


# In[228]:


newM1


# In[229]:


m0


# ### Using Spearman Correlation Coefficient 

# In[230]:


# calculate pearson correlation coefficient


geneCorr = []
for idx,gene in enumerate(m0.columns):
    if m0.iloc[:,idx].nunique() == 1 or newM1.iloc[:,idx].nunique() == 1:
        continue
    else:
        
        corrCoeff, pVal = pearsonr(m0.iloc[:,idx], newM1.iloc[:,idx])
        geneCorr.append((corrCoeff,pVal, gene))


# In[231]:


geneCorr


# In[232]:


correlationList = [abs(i[0]) for i in geneCorr if not math.isnan(i[0])]


# In[233]:


geneList = [i[2] for i in geneCorr]


# In[234]:


correlationList


# In[235]:


# sort the values in the list 
# create a list with the sorted values 
# create a list with the indices of sorted values


# In[236]:


indicesSort = sorted(range(len(correlationList)), key=lambda ind : correlationList[ind])


# In[237]:


corrSort = [correlationList[ind] for ind in indicesSort]


# In[238]:


geneSort = [geneList[ind] for ind in indicesSort]


# In[239]:


# sorting the indices based on descending order of correlation coefficient
# sorting correlation coefficient in descending order
# sorting the genes according to descending correlation coefficient value

indicesSortDesc = indicesSort[::-1]
corrSortDesc = corrSort[::-1]
geneSortDesc = geneSort[::-1]


# In[240]:


corrSortDesc 


# In[241]:


geneSortDesc


# In[242]:


# export this as a csv file

corrTable = pd.DataFrame({'Index': indicesSortDesc, 
                         'Correlation Coefficient': corrSortDesc,
                         'Genes': geneSortDesc})

corrTable.to_csv('kidneyClearCellPanCancerFinalRNASeq.csv', index=False)


# ### Using leave-one-out cross validation to get accuracy score for Logistic Regression

# In[243]:


m1Data = newM1.loc[:,geneSortDesc[:200]]
m1Data['type'] = 1
m1Data


# In[244]:


m0Data = m0.loc[:,geneSortDesc[:200]]
m0Data['type'] = 0
m0Data


# In[245]:


finalData = pd.concat([m0Data, m1Data], axis=0)
finalData


# In[ ]:





# In[246]:


filteredDataType = finalData.reset_index(drop=True)


# In[247]:


filteredDataType.reset_index(drop=True)


# ### Using Logistic Regression

# In[248]:




finalAcc = []


for g in range(40):
    train = filteredDataType


    predictionsLr = []
    actual = []
    
    for sampleID in range(802):

        trainCV = train.copy()
   

        testData = trainCV.iloc[sampleID]
        trainData = trainCV.drop(sampleID)
        k = (g+5)
        
        scaler = StandardScaler()
        
        trX = trainData.iloc[:,:k].values
        trY = trainData.iloc[:,-1].values

        teX = testData.iloc[:k].values.reshape(1,-1)
        teY = testData.iloc[-1]
        
        trXScaled = scaler.fit_transform(trX)
        teXScaled = scaler.transform(teX)

        lr = LogisticRegression(max_iter=10000)
        lr.fit(trXScaled, trY)
        yPredLr = lr.predict(teXScaled)
        predictionsLr.append(yPredLr)
        actual.append(teY)
  
    #print(predictionsLr)
    pLr = accuracy_score(predictionsLr, actual)
    finalAcc.append(pLr)
    print(pLr)


# In[249]:


finalAcc


# In[250]:


plt.figure(figsize=(10, 6)) 

genes = [i*5 for i in range(40)]

plt.errorbar(genes,finalAcc,linestyle='-', marker='.', color='blue')

plt.title("Kidney Clear Cell Pan Cancer RNASeq Data")


p1 = mlines.Line2D([], [], color='blue', marker='.', linestyle='-',
                          markersize=10, label='Accuracy for Logistic Regression (Correlation Coefficient)')


plt.legend(handles=[p1])




plt.xlabel("Number of Genes",size =15)
plt.ylabel("Accuracy Value",size=15)


# In[ ]:




