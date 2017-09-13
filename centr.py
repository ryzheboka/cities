import pandas as pd
import numpy as np
from time import time
import math

def define_features(line): ###für einen gegebenen Namen features zurückgeben
    features=list()

    #feature1
    for i in [2,3,4]:
        end=line[0][-i:]
        features.append(end)
    return features
    ###Inplement more features

def data_to_features(line,all_features): #create a list where 1 means the feature with this index is correct for the given name(der Rest ist 0)
    features_l=[0]*len(all_features)
    for i in [2,3,4]:
        end=line[0][-i:]
        features_l[all_features.index(end)]=1
        #print(len(features_l))
    return pd.Series(features_l)
   #print(features_l)

def sigmoid(z):
    return 1/(1+math.e-z)

def cost_function(predictions,y):
    m = len(predictions)

def pred_to_boolin(answers): #die Labelliste in binäre Form bringen
    answers[answers == "US"] = 1
    answers[answers == "CN"] = 0
    return answers

def gradient(predictions,y):
    pass

train_data = pd.read_csv("dataset/two_countries/train", sep="#",header=None)
print(train_data.head())

###turn the data into matrix X
time1=time()
all_features = train_data.apply(define_features,axis=1) ###create list of endings for each name
all_features = all_features.sum() ### sum all the lists to one
all_features = list(set(all_features)) ### remove duplicates
print(all_features)
print(time()-time1)
time2=time()
matr_x = train_data.apply(lambda x: data_to_features(x, all_features),axis=1) #for each name: either a feature is in the name or not
print(time()-time2)
print(matr_x.head())
print(len(all_features))
#Logistic regression(minimize cost_function)
