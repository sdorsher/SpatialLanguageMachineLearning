#!/usr/bin/python3
import os, sys

from sklearn.preprocessing import StandardScaler
import matplotlib
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



#what I want is the histogram of counts within each country to compare distributions between countries by eye. I can then look at a regression within the US by longitude and lattitude (actual distance from Chicago?) if it looks interesting

ratio=[]
coords=[]
alldat=[]

for line in file:
    line=re.split(',',line);
    thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
    ratio.append([thisratioV,thisratioS,thisratioO])
    coords.append([float(line[6]),float(line[7])])
    alldat.append([thisratioV,thisratioS,thisratioO,float(line[6]),float(line[7])])

ratioarray=np.asarray(ratio)
coordsarray=np.asarray(coords)
alldatarray=np.asarray(alldat)
        
print ratioarray[1:3]
print alldatarray[1:3]

scaler = StandardScaler()
scaler.fit(alldatarray)
StandardScaler(copy=True, with_mean=True, with_std=True)
trainingdata=scaler.transform(alldatarray)

km=KMeans(n_clusters=15,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(trainingdata)




print y_km




fig=plt.figure()
ax=fig.gca()

ax.scatter(coordsarray[:,1],coordsarray[:,0],c=y_km,cmap=cm.plasma_r)


plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('World map of language clusters')
fig.savefig('WorldMapLangClustering.png')
