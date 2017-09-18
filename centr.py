import pandas as pd
import numpy as np
from time import time
import math
import scipy.optimize as op
import os



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
        if end in (all_features):
            features_l[all_features.index(end)]=1
        #print(len(features_l))
    return pd.Series(features_l)
   #print(features_l)

def sigmoid(z):
    #print(float(1.0)+math.e**-z)
    s = 1/(1+math.exp(-z))
    if s == 1:
        #print("WRONG")
        s=1-1/1000000000000000
    return s

def cost_function(theta,x,ys):
    global min_cost
    global optimal_theta

    m, n = x.shape
    theta = theta.reshape((n, 1))
    ys = ys.values.reshape((m, 1))
    x = x.values.reshape((m, n))
    pred = [0] * m
    cost=0

    for i in range(0, m):
        new_x = x[i][:]
        pred[i] = sigmoid(new_x.dot(theta))
        if pred[i]>0 and pred[i]<1:
            pass
        else:
            print(pred[i])
        math.log(1 - pred[i])
        cost += -ys[i]*math.log(pred[i])-(1-ys[i])*math.log(1-pred[i]) #vectorized version would be better

    if float(cost/m) < min_cost:
        optimal_theta = theta
        min_cost = cost/m
        print("huraa")
        print("Optimales Theta"+optimal_theta)
    else:
        print(float(cost/m),min_cost)
    print(theta)
    print(cost/m)

    return float(cost/m)


def answ_to_boolean(answers): #die Labelliste in binäre Form bringen
    answers[answers == "US"] = 1
    answers[answers == "CN"] = 0
    return answers

def myRound(n):
    if n >=  0.5:
        return math.floor(n+1)
    else:
        return math.floor(n)

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
        grad=1/m*0.03*sum(dif_p_y.T.dot(x.T[:][j]))
        theta[j]=grad
    return theta

def accuracy(theta,x,y):
    m, n = x.shape
    theta = theta.reshape((n, 1))
    print(theta)
    y = y.values.reshape((m, 1))
    x = x.values.reshape((m, n))
    pred = [0] * m
    score=0
    for i in range(0, m):
        new_x = x[i][:]
        pred[i] = sigmoid(new_x.dot(theta))  # compute predictions
        #print(pred[i])
        rounded=myRound(pred[i])
        #print((pred))
        if rounded==y[i]:
            #print("Hurraaa")
            score+=1
    return score/m


train_data = pd.read_csv("dataset/two_countries/train", sep="#",header=None)

###turn the data into matrix X
time1=time()
all_features = train_data.apply(define_features,axis=1) ###create list of endings for each name
all_features = all_features.sum() ### sum all the lists to one
all_features = list(set(all_features)) ### remove duplicates



matr_x = train_data.apply(lambda x: data_to_features(x, all_features),axis=1) #for each name: either a feature is in the name or not
m,n=matr_x.shape
matr_x.insert(loc=0,column=n,value=pd.Series([1]*m).T)
m,n=matr_x.shape
init_theta=np.array([0.0]*n).T
y=answ_to_boolean(train_data[1])
min_cost=1
#optimal_theta=init_theta

Result = op.minimize(fun = cost_function,
                     x0 = init_theta,
                     args = (matr_x, y),
                     method = 'TNC',
                     jac = gradient,
                     )

#optimal_theta = Result.x
#print(Result.success)
optimal_theta = pd.Series([float(optimal_theta[i]) for i in range(0,len(optimal_theta)-1)])
print(optimal_theta)
#o_theta = optimal_theta.reshape((n, 1))


here = os.path.abspath(os.path.dirname(__file__))
optimal_theta.to_csv(here+"/theta_2C", sep='#',index=False, header=False)


###train accuracy
print("STOP")
print(cost_function(optimal_theta,matr_x,y))
train_accuracy=accuracy(optimal_theta,matr_x,y)
print(train_accuracy)

###valid accuracy
# ...
# ...
# ...

"""name=input("city: ")
features=data_to_features(name,all_features).append(1).reshape(1,n)
print(features)
label=sigmoid(data_to_features(name,all_features).dot(o_theta))
if abs(label)==1:
    print("USA")
else:
    print("China")"""