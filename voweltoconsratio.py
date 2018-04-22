#!/usr/bin/python
import os, sys



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
    #put a one at the end to account for one airport in number totaled. will be normalized later

    return vowels/consonents

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
    
#    if line[3] == "\"France\"":
#        thisFranceratio = VowelToConsRatio(line[1])
#        Franceratio.append(thisFranceratio)
    
    
#    if line[3] == "\"Germany\"":#
#        thisGermanyratio = VowelToConsRatio(line[1])
#        Germanyratio.append(thisGermanyratio)
    
#    if line[3] == "\"Mexico\"" :
#        thisMexicoratio = VowelToConsRatio(line[1])
#        Mexicoratio.append(thisMexicoratio)
    if line[3] == "\"Canada\"" :
        thisCanadaratio = VowelToConsRatio(line[1])
        Canadaratio.append(thisCanadaratio)
        
                




fig, ax = plt.subplots()
n_groups=len(Canadaratio)
index=np.arange(n_groups)
bar_width=0.9
opacity = 0.8
rects2 = plt.bar(index, np.asarray(Canadaratio), bar_width, alpha=opacity, color='g', label='Vowels/Consonants ratio')



plt.xlabel('Airports in Canada')
plt.ylabel('Ratio of letters in Airport names in Canada')
plt.legend()

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()
n_groups=len(USAratio)
index=np.arange(n_groups)
bar_width=0.9
opacity = 0.8
rects2 = plt.bar(index, np.asarray(USAratio), bar_width, alpha=opacity, color='g', label='Vowels/Consonants ratio')


plt.xlabel('Airports in the USA')
plt.ylabel('Letter ratio in Airport names in the USA')
plt.legend()

plt.tight_layout()
plt.show()
fig, ax = plt.subplots()
n_groups=len(Englandratio)
index=np.arange(n_groups)
bar_width=0.9
opacity = 0.8
rects2 = plt.bar(index, np.asarray(Englandratio), bar_width, alpha=opacity, color='g', label='Vowels/Consonants ratio')



plt.xlabel('Airports in the UK')
plt.ylabel('Letter ratio in Airport names in the UK')
plt.legend()

plt.tight_layout()
plt.show()


