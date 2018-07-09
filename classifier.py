#!/usr/bin/python
import os, sys

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import cauchy
import numpy as np
from io import StringIO
import re
from sklearn.model_selection import train_test_split

file = open("airports-extended.dat.txt","r")

def powerlaw(x,a,b,c):
    return a*x**b+c

def f5(x,a,b,c,d,g,h):
    return a*x**5+b*x**4+c*x**3+d*x**2+g*x+h

def f3(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

def VowelToConsRatio(name):
    
    words = sum(c.isalpha() for c in name)
    spaces = sum(c.isspace() for c in name)
    others = len(name) - words - spaces
    vowels = sum(map(name.lower().count, "aeiou"))
    consonents = words - vowels


    return float(vowels)/float(consonents), float(spaces)/float(consonents), float(others)/float(consonents)



def AirportNameLetterFrequency(name):
    
    words = sum(c.isalpha() for c in name)
    spaces = sum(c.isspace() for c in name)
    others = len(name) - words - spaces
    vowels = sum(map(name.lower().count, "aeiou"))
    consonents = words - vowels
    #put a one at the end to account for one airport in number totaled. will be normalized later

    return [consonents, vowels, spaces, others, 1]


def MergeCount(totalCount, thisCount):
    totalCount2=totalCount+thisCount
    return totalCount2



def LatLongClassMaker(coor):
    lat,long=coor
    if lat<30:
        if long<90:
            return 1 #florida
        elif long > 120:
            return 2 #texas
        else:
            return 3 #hawaii
    elif lat >50:
        return 4 #alaska
    elif long>80:
        return (50-lat)/20*5*(125-long)/45*9+7
    elif long<80 and lat < 40:
        return 5 #south east
    elif long>80 and lat > 40:
        return 6 #new england
    #worst mapping ever-- ignores new england and city diversity

ratio=[]
coords=[]


for line in file:
    line=re.split(',',line);

    if line[3] == "\"United States\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])


ratioarray=np.empty([len(ratio),3])
coordsarray=np.empty([len(coords),2])

for i, rat in enumerate(ratio):
    ratioarray[i,:]=rat
for i,coor in enumerate(coords):
    coordsarray[i,:]=LatLongClassMaker(coor)

print(len(ratioarray))
print(len(coordsarray))        
print(ratioarray[1:3])
print(coordsarray[1:3])


xtrain,xtest,ytrain,ytest=train_test_split(ratioarray,coordsarray,test_size=0.33,shuffle=True, random_state=42)

scaler = StandardScaler()
xtrain2=scaler.fit_transform(xtrain)





#clf = SGDClassifier(loss='modified_huber', penalty='l2')
#clf.fit(trainingdat,coords)
#modified_huber loss function (counting values between 0 and 1  to continuous values), using l2 (RSS) error for distance measure. note that this does not make use of power law scaling

clf=OneVersusRestClassifier(random_state=None,shuffle=True, tol=1.e-2)

clf.fit(xtrain2,ytrain)
predictions=clf.predict(xtrain2)
distance=clf.decision_function(xtrain2)

accuracy=accuracy_score(ytest,prediction)
print(accuracy)

fig=plt.figure()
ax=fig.gca()


ax.scatter(xtrain2[:,1],xtrain2[:,0],c=predictions,cmap=cm.plasma_r)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Map of language classes for US using extended training set')
plt.show()


