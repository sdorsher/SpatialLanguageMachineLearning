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

def VowelToConsRatio(name):
    
    words = sum(c.isalpha() for c in name)
    spaces = sum(c.isspace() for c in name)
    others = len(name) - words - spaces
    vowels = sum(map(name.lower().count, "aeiou"))
    consonents = words - vowels


    return float(vowels)/float(consonents)


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

Chinaratio=[]
USAratio=[]
Englandratio=[]
Germanyratio=[]
Franceratio=[]
Canadaratio=[]
Mexicoratio=[]

for line in file:
    line=re.split(',',line);
#    if line[3] == "\"China\"":
#        thisChinaratio = VowelToConsRatio(line[1])
#        Chinaratio.append(thisChinaratio)
    if line[3] == "\"United States\"":
        thisUSAratio = VowelToConsRatio(line[1])
        USAratio.append(thisUSAratio)
    if line[3] == "\"United Kingdom\"":
        thisEnglandratio = VowelToConsRatio(line[1])
        Englandratio.append(thisEnglandratio)
    
    if line[3] == "\"France\"":
        thisFranceratio = VowelToConsRatio(line[1])
        Franceratio.append(thisFranceratio)
    
    
    if line[3] == "\"Germany\"":#
        thisGermanyratio = VowelToConsRatio(line[1])
        Germanyratio.append(thisGermanyratio)
    
    if line[3] == "\"Mexico\"" :
        thisMexicoratio = VowelToConsRatio(line[1])
        Mexicoratio.append(thisMexicoratio)
    if line[3] == "\"Canada\"" :
        thisCanadaratio = VowelToConsRatio(line[1])
        Canadaratio.append(thisCanadaratio)
        
                
Canadaarr=np.asarray(Canadaratio)
Canadaarr.sort()
USAarr=np.asarray(USAratio)
USAarr.sort()
UKarr=np.asarray(Englandratio)
UKarr.sort()
Francearr=np.asarray(Franceratio)
Francearr.sort()
Germanyarr=np.asarray(Germanyratio)
Germanyarr.sort()
Mexicoarr=np.asarray(Mexicoratio)
Mexicoarr.sort()



fig, ax = plt.subplots()
n_groups=len(USAratio)
index=np.arange(n_groups)
bar_width=0.9
opacity = 0.8
rects1 = plt.bar(index, USAarr[::-1], bar_width, alpha=opacity, color='b', label='USA')


plt.xlabel('Airports in the USA')
plt.ylabel('Letter ratio in Airport names in the USA')
plt.title('Vowels/Consonants ratio, sorted')

plt.tight_layout()
plt.show()




fig, ax = plt.subplots()
plt.xlabel('Airports in the USA')
plt.ylabel('Letter ratio in Airport names')
plt.title('Vowels/Consonants ratio, sorted')
points1 = plt.plot(USAarr[::-1],color='b', label='USA')
points2 = plt.plot(Canadaarr[::-1],color='g', label='Canada')
points3 = plt.plot(UKarr[::-1],color='r', label='UK')
points4 = plt.plot(Francearr[::-1],color='c', label='France')
points5 = plt.plot(Germanyarr[::-1],color='m', label='Germany')
points6 = plt.plot(Mexicoarr[::-1],color='y', label='Mexico')
plt.legend()
plt.show()

#falls off continuously until there is one. 
#this looks like a breit-wigner distribution
#also known as cauchy or lorentz distribution
#try doing a fit of the cauchy distribution with an x intercept of zero to the usa data
#try my first thought, that I got sidetracked from by a paper I saw today. log(gaussian)


xdat=np.arange(len(USAarr))
ydat=USAarr[::-1]



fig=plt.subplots()
plot1=plt.plot(xdat, cauchy.pdf(xdat,0,400),color='g',label='Cauchy distribution')
plot2=plt.plot(xdat,ydat,color='b',label='data')
plt.xlabel("Airports")
plt.ylabel("Verb to Consonant ratio in airport name")
plt.title("USA data")
plt.show()
