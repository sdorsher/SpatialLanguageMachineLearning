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


file = open("airports.dat.txt","r")

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

#def CounttoHistograms(thisUSAcount, totalUSAcount, totalEnglandcount, totalTurkeycount, totalFrancecount, totalGermanycount, totalMexicocount, totalCanadacount)




totalUSAcount=np.zeros(5)
totalChinacount=np.zeros(5)
totalEnglandcount=np.zeros(5)
totalFrancecount=np.zeros(5)
totalGermanycount=np.zeros(5)
totalMexicocount=np.zeros(5)
totalCanadacount=np.zeros(5)



#I picked some countries I have familiarity with so I know whether or not the data makes sense when I see it, sort of.

#what I want is the histogram of counts within each country to compare distributions between countries by eye. I can then look at a regression within the US by longitude and lattitude (actual distance from Chicago?) if it looks interesting

ratio=[]
coords=[]


for line in file:
    line=re.split(',',line);

    if line[3] == "\"China\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
    if line[3] == "\"United States\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
    if line[3] == "\"United Kingdom\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
    if line[3] == "\"France\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
    
    if line[3] == "\"Germany\"":#
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
    if line[3] == "\"Mexico\"" :
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
    if line[3] == "\"Canada\"" :
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])

ratioarray=np.asarray(ratio)
coordsarray=np.asarray(coords)
        
print ratioarray[1:3]
print coords[1:3]

scaler = StandardScaler()
scaler.fit(ratio)
StandardScaler(copy=True, with_mean=True, with_std=True)

trainingdat=scaler.transform(ratioarray)



clf = SGDClassifier(loss='modified_huber', penalty='l2')
clf.fit(trainingdat,coords)
#modified_huber loss function (counting values between 0 and 1  to continuous values), using l2 (RSS) error for distance measure. note that this does not make use of power law scaling

SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', max_iter=None, n_iter=None,
       n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
       shuffle=True, tol=None, verbose=0, warm_start=False

classlabels=clf.predict(trainingdata)
distance=clf.decision_function(trainingdata)
print len(z)
print trainingdat.shape
print coords.shape

#testphraseV,testphraseS,testphraseO=VowelToConsCount("Hello World!")
#testloc=clf.predict([[testphraseV,testphraseS,testphraseO]])
#print testloc


fig=plt.figure()
ax=fig.gca()


ax.scatter(trainingdata[:,1],trainingdata[:,0],c=classlabels,cmap=cm.plasma_r)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Map of classes in language space for seven countries')
plt.show()



fig=plt.figure()
ax=fig.gca()

ax.scatter(coords[:,1],coords[:,0],c=classlabels,cmap=cm.plasma_r)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('World map of language classes for seven countries')
plt.show()


fig=plt.figure()
ax=fig.gca()


ax.scatter(coords[:,1],coords[:,0],c=distance,cmap=cm.plasma_r)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('World map of distances from the centers of language classes')
plt.show()
