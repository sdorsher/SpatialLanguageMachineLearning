#!/usr/bin/python
import os, sys

from sklearn.preprocessing import StandardScaler
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
alldat=[]

for line in file:
    line=re.split(',',line);

    if line[3] == "\"China\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
        alldat.append([thisratioV,thisratioS,thisratioO,float(line[6]),float(line[7])])
    if line[3] == "\"United States\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
        alldat.append([thisratioV,thisratioS,thisratioO,float(line[6]),float(line[7])])
    if line[3] == "\"United Kingdom\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
        alldat.append([thisratioV,thisratioS,thisratioO,float(line[6]),float(line[7])])
    if line[3] == "\"France\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
        alldat.append([thisratioV,thisratioS,thisratioO,float(line[6]),float(line[7])])
    
    if line[3] == "\"Germany\"":#
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
        alldat.append([thisratioV,thisratioS,thisratioO,float(line[6]),float(line[7])])
    if line[3] == "\"Mexico\"" :
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])
        alldat.append([thisratioV,thisratioS,thisratioO,float(line[6]),float(line[7])])
    if line[3] == "\"Canada\"" :
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
trainingdat=scaler.transform(alldatarray)

km=KMeans(n_clusters=7,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_km=km.fit_predict(trainingdata)




print y_km


fig = plt.figure()
ax=fig.gca()


ax.scatter(ratioarray[:,0],ratioarray[:,1],c=y_km,cmap=cm.plasma_r)

plt.xlabel('Vowel/Consonant')
plt.ylabel('Space/Consonant')
plt.title('Clustering in Language Space, axis 1')
plt.show()


fig = plt.figure()
ax=fig.gca()

ax.scatter(ratioarray[:,0],ratioarray[:,2],c=y_km,cmap=cm.plasma_r)

plt.xlabel('Vowel/Consonant')
plt.ylabel('Other/Consonant')
plt.title('Clustering in Language Space, axis 2')
plt.show()


fig = plt.figure()
ax=fig.gca()

ax.scatter(ratioarray[:,1],ratioarray[:,2],c=y_km,cmap=cm.plasma_r)

plt.xlabel('Space/Consonant')
plt.ylabel('Other/Consonant')
plt.title('Clustering in Language Space, axis 3')
plt.show()


fig=plt.figure()
ax=fig.gca()

ax.scatter(coordsarray[:,1],coordsarray[:,0],c=y_km,cmap=cm.plasma_r)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('World map of language clusters for seven countries')
plt.show()
