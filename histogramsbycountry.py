#!/usr/bin/python
import os, sys



import numpy as np
from io import StringIO
import re
import matplotlib.pyplot as plt

file = open("airports.dat.txt","r")

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

Chinavowel=[]
Chinacons=[]
USvowel=[]
UScons=[]
Englandvowel=[]
Englandcons=[]
Francevowel=[]
Francecons=[]
Germanyvowel=[]
Germanycons=[]
Mexicovowel=[]
Mexicocons=[]
Canadavowel=[]
Canadacons=[]

for line in file:
    line=re.split(',',line);
    if line[3] == "\"China\"":
        thisChinacount = AirportNameLetterFrequency(line[1])
        Chinavowel.append(thisChinacount[1])
        Chinacons.append(thisChinacount[0])
    if line[3] == "\"United States\"":
        thisUSAcount = AirportNameLetterFrequency(line[1])
        USvowel.append(thisUSAcount[1])
        UScons.append(thisUSAcount[0])
    
    if line[3] == "\"United Kingdom\"":
        thisEnglandcount = AirportNameLetterFrequency(line[1])
        Englandvowel.append(thisEnglandcount[1])
        Englandcons.append(thisEnglandcount[0])
    
    if line[3] == "\"France\"":
        thisFrancecount = AirportNameLetterFrequency(line[1])
        Francevowel.append(thisFrancecount[1])
        Francecons.append(thisFrancecount[0])
    
    if line[3] == "\"Germany\"":
        thisGermanycount = AirportNameLetterFrequency(line[1])
        Germanyvowel.append(thisGermanycount[1])
        Germanycons.append(thisGermanycount[0])
    
    if line[3] == "\"Mexico\"" :
        thisMexicocount = AirportNameLetterFrequency(line[1])
        Mexicovowel.append(thisMexicocount[1])
        Mexicocons.append(thisMexicocount[0])
    if line[3] == "\"Canada\"" :
        thisCanadacount = AirportNameLetterFrequency(line[1])
        Canadavowel.append(thisCanadacount[1])
        Canadacons.append(thisCanadacount[0])
        
                




fig, ax = plt.subplots()
n_groups=len(Chinavowel)
index=np.arange(n_groups)
bar_width=0.35
opacity = 0.8
rects2 = plt.bar(index, np.asarray(Chinacons), bar_width, alpha=opacity, color='g', label='Consonants')
rects1 = plt.bar(index, np.asarray(Chinavowel), bar_width, alpha=opacity, color='b', label='Vowels')


plt.xlabel('Airports in China')
plt.ylabel('Letters in Airport names in China')
plt.legend()

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
n_groups=len(USvowel)
index=np.arange(n_groups)
bar_width=0.35
opacity = 0.8
rects2 = plt.bar(index, np.asarray(UScons), bar_width, alpha=opacity, color='g', label='Consonants')
rects1 = plt.bar(index, np.asarray(USvowel), bar_width, alpha=opacity, color='b', label='Vowels')


plt.xlabel('Airports in the USA')
plt.ylabel('Letters in Airport names in the USA')
plt.legend()

plt.tight_layout()
plt.show()
fig, ax = plt.subplots()
n_groups=len(Mexicovowel)
index=np.arange(n_groups)
bar_width=0.35
opacity = 0.8
rects2 = plt.bar(index, np.asarray(Mexicocons), bar_width, alpha=opacity, color='g', label='Consonants')
rects1 = plt.bar(index, np.asarray(Mexicovowel), bar_width, alpha=opacity, color='b', label='Vowels')


plt.xlabel('Airports in Mexico')
plt.ylabel('Letters in Airport names in Mexico')
plt.legend()

plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
n_groups=len(Germanyvowel)
index=np.arange(n_groups)
bar_width=0.35
opacity = 0.8
rects2 = plt.bar(index, np.asarray(Germanycons), bar_width, alpha=opacity, color='g', label='Consonants')
rects1 = plt.bar(index, np.asarray(Germanyvowel), bar_width, alpha=opacity, color='b', label='Vowels')


plt.xlabel('Airports in Germany')
plt.ylabel('Letters in Airport names in Germany')
plt.legend()

plt.tight_layout()
plt.show()
