import numpy as np
from io import StringIO
import re
import plot

file = open("airports.dat.txt","r")

def AirportNameLetterFrequency(name):
    #has to be typable on my keyboard. I'm going to have to figure this out. For prototyping, use 26 character alphabet with - and / and space make it case insensitive. I think this is the wrong answer and am going to contemplate this overnight.

#look at extended data tomorrow and see what can be done within a given country. starting with USA, maybe compare to canada and another country with a similar character set.

totalMexicoHist=Null
totalUSAhist=Null
totalChinaHist=Null 

for line in file:
    re.split(',',line);
    print line
    if line(2) == "Mexico": 
        thisMexicohist=AirportNameLetterFrequency(line(1))
        MergeHistograms(totalMexicoHist,thisMexicohist)
    if line(2) == "United States":
        thisUSAhist = AirportNameLetterFrequency(line(1))
        MergeHistograms(totalUSAhist,thisUSAhist)
    if line(2) == "China":
        thisChinahist = AirportNameLetterFrequency(line(1))
        MergeHistograms(totalChinahist,thisChinahist)


plot.
