import pandas as pd
import numpy as np
from time import time
import math


def define_features(line): ###für einen gegebenen Namen features zurückgeben
    features=list()

    #feature1
    for i in [3]:
        end=line[0][-i:]
        features.append(end)
    return features
    ###Inplement more features

def data_to_features(line,all_features): #create a list where 1 means the feature with this index is correct for the given name(der Rest ist 0)
    features_l=[0]*len(all_features)
    for i in [3]:
        end=line[0][-i:]
        features_l[all_features.index(end)]=1
        #print(len(features_l))
    return pd.Series(features_l)
   #print(features_l)

def sigmoid(z):
    return 1/(1+math.e-z)

def cost_function(predictions,ys):
    m = len(predictions)
    cost=0
    time3 = time()
    for i in range(m):
        cost+=-ys[i]*math.log(predictions[i])-(1-ys[i])*math.log(1-predictions[i]) #vectorized version would be better
    print(time() - time3)
    return cost/m


def answ_to_boolean(answers): #die Labelliste in binäre Form bringen
    answers[answers == "US"] = 1
    answers[answers == "CN"] = 0
    return answers

def gradient(theta,x,y):
    m,n = x.shape
    theta = theta.reshape((n, 1))
    y = y.values.reshape((m, 1))
    x = x.values.reshape((m,n))
    #print(x.head())
    pred=[0]*m
    dif_p_y=list()
    for i in range(0,m):
        new_x=x[i][:]
        pred[i]=sigmoid(new_x.dot(theta)) #compute predictions
        dif_p_y.append(pred[i]-y[i])
    dif_p_y=np.array(dif_p_y)
    for j in range(n):
        #print(x.T[:][j])
        grad=1/m*sum(dif_p_y.T.dot(x.T[:][j]))
        theta[j]=grad
    return theta




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
#matr_x=matr_x.append([1]*matr_x.shape[1])
m,n=matr_x.shape
print(m)
matr_x.insert(loc=0,column=n,value=pd.Series([1]*m).T)
#matr_x=np.array(([[1]*m,matr_x])).reshape(m,n+1)
print(time()-time2)
print(matr_x.head())
#print(matr_x.iloc[:][:])
m,n=matr_x.shape
print(m,n)
init_theta=np.array([0.0]*n).T
y=answ_to_boolean(train_data[1])
print(gradient(init_theta,matr_x,y))

#Logistic regression(minimize cost_function)"""
