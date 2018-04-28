#!/usr/bin/python
import os, sys

from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import cauchy
import numpy as np
from io import StringIO
import re
import matplotlib.pyplot as plt

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

ChinaratioV=[]
USAratioV=[]
EnglandratioV=[]
GermanyratioV=[]
FranceratioV=[]
CanadaratioV=[]
MexicoratioV=[]
ChinaratioS=[]
USAratioS=[]
EnglandratioS=[]
GermanyratioS=[]
FranceratioS=[]
CanadaratioS=[]
MexicoratioS=[]
ChinaratioO=[]
USAratioO=[]
EnglandratioO=[]
GermanyratioO=[]
FranceratioO=[]
CanadaratioO=[]
MexicoratioO=[]

for line in file:
    line=re.split(',',line);
    if line[3] == "\"China\"":
        thisChinaratioV, thisChinaratioS, thisChinaratioO = VowelToConsRatio(line[1])
        ChinaratioV.append(thisChinaratioV)
        ChinaratioS.append(thisChinaratioS)
        ChinaratioO.append(thisChinaratioO)
    if line[3] == "\"United States\"":
        thisUSAratioV, thisUSAratioS, thisUSAratioO = VowelToConsRatio(line[1])
        USAratioV.append(thisUSAratioV)
        USAratioS.append(thisUSAratioS)
        USAratioO.append(thisUSAratioO)
    if line[3] == "\"United Kingdom\"":
        thisEnglandratioV, thisEnglandratioS, thisEnglandratioO = VowelToConsRatio(line[1])
        EnglandratioV.append(thisEnglandratioV)
        EnglandratioS.append(thisEnglandratioS)
        EnglandratioO.append(thisEnglandratioO)
    
    if line[3] == "\"France\"":
        thisFranceratioV, thisFranceratioS, thisFranceratioO = VowelToConsRatio(line[1])
        FranceratioV.append(thisFranceratioV)
        FranceratioS.append(thisFranceratioS)
        FranceratioO.append(thisFranceratioO)
    
    
    if line[3] == "\"Germany\"":#
        thisGermanyratioV, thisGermanyratioS, thisGermanyratioO = VowelToConsRatio(line[1])
        GermanyratioV.append(thisGermanyratioV)
        GermanyratioS.append(thisGermanyratioS)
        GermanyratioO.append(thisGermanyratioO)
    
    if line[3] == "\"Mexico\"" :
        thisMexicoratioV, thisMexicoratioS, thisMexicoratioO = VowelToConsRatio(line[1])
        MexicoratioV.append(thisMexicoratioV)
        MexicoratioS.append(thisMexicoratioS)
        MexicoratioO.append(thisMexicoratioO)
    if line[3] == "\"Canada\"" :
        thisCanadaratioV, thisCanadaratioS, thisCanadaratioO = VowelToConsRatio(line[1])
        CanadaratioV.append(thisCanadaratioV)
        CanadaratioS.append(thisCanadaratioS)
        CanadaratioO.append(thisCanadaratioO)
        
ChinaarrV=np.asarray(ChinaratioV)
ChinaarrV.sort()                
CanadaarrV=np.asarray(CanadaratioV)
CanadaarrV.sort()
USAarrV=np.asarray(USAratioV)
USAarrV.sort()
UKarrV=np.asarray(EnglandratioV)
UKarrV.sort()
FrancearrV=np.asarray(FranceratioV)
FrancearrV.sort()
GermanyarrV=np.asarray(GermanyratioV)
GermanyarrV.sort()
MexicoarrV=np.asarray(MexicoratioV)
MexicoarrV.sort()
ChinaarrS=np.asarray(ChinaratioS)
ChinaarrS.sort()
CanadaarrS=np.asarray(CanadaratioS)
CanadaarrS.sort()
USAarrS=np.asarray(USAratioS)
USAarrS.sort()
UKarrS=np.asarray(EnglandratioS)
UKarrS.sort()
FrancearrS=np.asarray(FranceratioS)
FrancearrS.sort()
GermanyarrS=np.asarray(GermanyratioS)
GermanyarrS.sort()
MexicoarrS=np.asarray(MexicoratioS)
MexicoarrS.sort()
ChinaarrO=np.asarray(ChinaratioO)
ChinaarrO.sort()
CanadaarrO=np.asarray(CanadaratioO)
CanadaarrO.sort()
USAarrO=np.asarray(USAratioO)
USAarrO.sort()
UKarrO=np.asarray(EnglandratioO)
UKarrO.sort()
FrancearrO=np.asarray(FranceratioO)
FrancearrO.sort()
GermanyarrO=np.asarray(GermanyratioO)
GermanyarrO.sort()
MexicoarrO=np.asarray(MexicoratioO)
MexicoarrO.sort()



fig, ax = plt.subplots()
n_groups=len(USAratioV)
index=np.arange(n_groups)
bar_width=0.9
opacity = 0.8
rects1 = plt.bar(index, USAarrV[::-1], bar_width, alpha=opacity, color='b', label='USA')


plt.xlabel('Airports in the USA')
plt.ylabel('Letter ratio in Airport names in the USA')
plt.title('Vowels/Consonants ratio, sorted')

plt.tight_layout()
plt.show()




fig, ax = plt.subplots()
n_groups=len(USAratioS)
index=np.arange(n_groups)
bar_width=0.9
opacity = 0.8
rects1 = plt.bar(index, USAarrS[::-1], bar_width, alpha=opacity, color='b', label='USA')


plt.xlabel('Airports in the USA')
plt.ylabel('Letter ratio in Airport names in the USA')
plt.title('Space/Consonants ratio, sorted')

plt.tight_layout()
plt.show()




fig, ax = plt.subplots()
n_groups=len(USAratioO)
index=np.arange(n_groups)
bar_width=0.9
opacity = 0.8
rects1 = plt.bar(index, USAarrO[::-1], bar_width, alpha=opacity, color='b', label='USA')


plt.xlabel('Airports in the USA')
plt.ylabel('Letter ratio in Airport names in the USA')
plt.title('Other Symbols/Consonants ratio, sorted')

plt.tight_layout()
plt.show()




fig, ax = plt.subplots()
plt.xlabel('Airports in the USA')
plt.ylabel('Letter ratio in Airport names')
plt.title('Vowels/Consonants ratio, sorted')
points1 = plt.plot(USAarrV[::-1],color='b', label='USA')
points2 = plt.plot(CanadaarrV[::-1],color='g', label='Canada')
points3 = plt.plot(UKarrV[::-1],color='r', label='UK')
points4 = plt.plot(FrancearrV[::-1],color='c', label='France')
points5 = plt.plot(GermanyarrV[::-1],color='m', label='Germany')
points6 = plt.plot(MexicoarrV[::-1],color='y', label='Mexico')
points7 = plt.plot(ChinaarrV[::-1],color='k',label='China')
plt.legend()
plt.show()


fig, ax = plt.subplots()
plt.xlabel('Airports in the USA')
plt.ylabel('Letter ratio in Airport names')
plt.title('Space/Consonants ratio, sorted')
points1 = plt.plot(USAarrS[::-1],color='b', label='USA')
points2 = plt.plot(CanadaarrS[::-1],color='g', label='Canada')
points3 = plt.plot(UKarrS[::-1],color='r', label='UK')
points4 = plt.plot(FrancearrS[::-1],color='c', label='France')
points5 = plt.plot(GermanyarrS[::-1],color='m', label='Germany')
points6 = plt.plot(MexicoarrS[::-1],color='y', label='Mexico')
points7 = plt.plot(ChinaarrS[::-1],color='k',label='China')
plt.legend()
plt.show()


fig, ax = plt.subplots()
plt.xlabel('Airports in the USA')
plt.ylabel('Letter ratio in Airport names')
plt.title('Other Symbol/Consonants ratio, sorted')
points1 = plt.plot(USAarrO[::-1],color='b', label='USA')
points2 = plt.plot(CanadaarrO[::-1],color='g', label='Canada')
points3 = plt.plot(UKarrO[::-1],color='r', label='UK')
points4 = plt.plot(FrancearrO[::-1],color='c', label='France')
points5 = plt.plot(GermanyarrO[::-1],color='m', label='Germany')
points6 = plt.plot(MexicoarrO[::-1],color='y', label='Mexico')
points7 = plt.plot(ChinaarrO[::-1],color='k',label='China')
plt.legend()
plt.show()

#falls off continuously until there is one. 
#this looks like a breit-wigner distribution
#also known as cauchy or lorentz distribution
#try doing a fit of the cauchy distribution with an x intercept of zero to the usa data
#try my first thought, that I got sidetracked from by a paper I saw today. log(gaussian)
#better yet try a third order polynomial

xdat=np.arange(len(USAarrV))
ydat=USAarrV[::-1]
par=[1.,-1.,0.]
p0=par
par,pvar=curve_fit(powerlaw,xdat,ydat)


fig=plt.subplots()
plot2=plt.plot(xdat,ydat,color='b',label='data')
plot1=plt.plot(xdat, powerlaw(xdat, par[0], par[1], par[2]),color='r',label='Power law')
plt.xlabel("Airports")
plt.ylabel("Vowel to Consonant ratio in airport name")
plt.title("USA data")
plt.legend()
plt.show()


xdatS=np.arange(len(USAarrS))
ydatS=USAarrS[::-1]
parS=[1.,-1.,0.]
p0S=parS
parS,pvarS=curve_fit(powerlaw,xdatS,ydatS)


fig=plt.subplots()
plot2=plt.plot(xdatS,ydatS,color='b',label='data')
plot1=plt.plot(xdatS, powerlaw(xdatS, parS[0], parS[1], parS[2]),color='r',label='Power law')
plt.xlabel("Airports")
plt.ylabel("Space to Consonant ratio in airport name")
plt.title("USA data")
plt.legend()
plt.show()


xdatO=np.arange(len(USAarrO))
ydatO=USAarrO[::-1]
parO=[1.,-1.,0.]
p0O=parO
parO,pvarO=curve_fit(powerlaw,xdatO,ydatO)


fig=plt.subplots()
plot2=plt.plot(xdatO,ydatO,color='b',label='data')
plot1=plt.plot(xdatO, powerlaw(xdatO, parO[0], parO[1], parO[2]),color='r',label='Power law')
plt.xlabel("Airports")
plt.ylabel("Other Symbol to Consonant ratio in airport name")
plt.title("USA data")
plt.legend()
plt.show()

#fits third order polynomial perfectly
