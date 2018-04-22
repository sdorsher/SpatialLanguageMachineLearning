#!/usr/bin/python

#not feasible due to lack of linguistic knowledge. 




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
totalTurkeycount=np.zeros(5)
totalFrancecount=np.zeros(5)
totalGermanycount=np.zeros(5)
totalMexicocount=np.zeros(5)
totalCanadacount=np.zeros(5)



#I picked some countries I have familiarity with so I know whether or not the data makes sense when I see it, sort of.

for line in file:
    line=re.split(',',line);
    if line[3] == "\"China\"":
        thisChinacount = AirportNameLetterFrequency(line[1])
        totalChinacount=MergeCount(totalChinacount,thisChinacount)
    if line[3] == "\"United States\"":
        thisUSAcount = AirportNameLetterFrequency(line[1])
        totalUSAcount= MergeCount(totalUSAcount,thisUSAcount)
    if line[3] == "\"United Kingdom\"":
        thisEnglandcount = AirportNameLetterFrequency(line[1])
        totalEnglandcount=MergeCount(totalEnglandcount,thisEnglandcount)
    if line[3] == "\"Turkey\"":
        thisTurkeycount = AirportNameLetterFrequency(line[1])
        totalTurkeycount=MergeCount(totalTurkeycount,thisTurkeycount)
    if line[3] == "\"France\"":
        thisFrancecount = AirportNameLetterFrequency(line[1])
        totalFrancecount=MergeCount(totalFrancecount,thisFrancecount)
    if line[3] == "\"Germany\"":
        thisGermanycount = AirportNameLetterFrequency(line[1])
        totalGermanycount=MergeCount(totalGermanycount,thisGermanycount)
    if line[3] == "\"Mexico\"" :
        thisMexicocount = AirportNameLetterFrequency(line[1])
        totalMexicocount=MergeCount(totalMexicocount,thisMexicocount)
    if line[3] == "\"Canada\"" :
        thisCanadacount = AirportNameLetterFrequency(line[1])
        totalCanadacount=MergeCount(totalCanadacount,thisCanadacount)
                
#ahist, ehist, ihist, ohist, uhist, yhist==CounttoHistograms(thisUSAcount, totalUSAcount, totalEnglandcount, totalTurkeycount, totalFrancecount, totalGermanycount, totalMexicocount, totalCanadacount)


print totalUSAcount/totalUSAcount[4]
print totalChinacount/totalChinacount[4]
print totalTurkeycount/totalTurkeycount[4]
print totalEnglandcount/totalEnglandcount[4]
print totalFrancecount/totalFrancecount[4]
print totalGermanycount/totalGermanycount[4]
print totalMexicocount/totalMexicocount[4]
print totalCanadacount/totalCanadacount[4]

#what I want is the histogram for china versus the histogram for mexico versus the histogram for the US.




#fig, ax = plt.subplots()
#index=np.arrange(n_groups)
#barwidth=0.25
#opacity = 0.8
#rects = plot.bar(index
