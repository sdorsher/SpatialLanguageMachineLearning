import numpy as np
from io import StringIO
import re
import matplotlib.pyplot as plt

file = open("airports.dat.txt","r")

def AirportNameLetterFrequency(name):
    
    #has to be typable on my keyboard. I'm going to have to figure this out. For prototyping, use 26 character alphabet with - and / and space make it case insensitive. I think this is the wrong answer and am going to contemplate this overnight.

#look at extended data tomorrow and see what can be done within a given country. starting with USA, maybe compare to canada and another country with a similar character set.


#for now, consider only the vowels
#M-x set-input-method latin-postfix
    atot=name.count('a')+name.count('á')+name.count('à')+name.count('â')+name.count('ä')+name.count('ą')+name.count('ã')+name.count('å')+name.count('ª') 

    etot=name.count('e')+name.count('é')+name.count('è')+name.count('ê')+name.count('ë')+name.count('ě')+name.count('ę')+name.count('ė') 

    itot= name.count('i')+name.count('í')+name.count('ì')+name.count('î')+name.count('í')+name.count('ĩ')+name.count('į')+name.count('ı')  

    otot=name.count('o')+name.count('ó')+name.count('ò')+name.count('ô')+name.count('ö')+name.count('õ')+name.count('ő')+name.count('ð')+name.count('º')+name.count('ø')  

    utot=name.count('u')+name.count('ú')+name.count('ù')+name.count('û')+name.count('ü')+name.count('ũ')+name.count('ų')+name.count('ű')+name.count('ů') 

    ytot=name.count('y')+name.count('ß')+name.count('w')+name.count('æ')+name.count('þ') 
    Vowels = [atot etot itot otot utot ytot]

    return Vowels


def MergeCount(totalCount, thisCount):
    totalCount=totalCount+thisCount
    return totalCount

def CounttoHistograms(thisUSAcount, totalUSAcount, totalEnglandcount, totalTurkeycount, totalFrancecount, totalGermanycount, totalMexicocount, totalCanadacount)




totalUSAcount=Null
totalChinacount=Null 

for line in file:
    re.split(',',line);
    print line
    if line(2) == "United States":
        thisUSAcount = AirportNameLetterFrequency(line(1))
        totalUSAcount= MergeCount(totalUSAcount,thisUSAcount)
    if line(2) == "China":
        thisChinacountt = AirportNameLetterFrequency(line(1))
        totalChinacount=MergeCount(totalChinacount,thisChinacount)
    if line(2) == "United Kingdom":
        thisEnglandcountt = AirportNameLetterFrequency(line(1))
        totalEnglandcount=MergeCount(totalEnglandcount,thisEnglandcount)
    if line(2) == "Turkey":
        thisTurkeycount = AirportNameLetterFrequency(line(1))
        totalTurkeycount=MergeCount(totalTurkeycount,thisTurkeycount)
 if line(2) == "France":
        thisFrancecountt = AirportNameLetterFrequency(line(1))
        totalFrancecount=MergeCount(totalFrancecount,thisFrancecount)
if line(2) == "Germany":
        thisGermanycountt = AirportNameLetterFrequency(line(1))
        totalGermanycount=MergeCount(totalGermanycount,thisGermanycount)
if line(2) == "Mexico" :
        thisMexicocountt = AirportNameLetterFrequency(line(1))
        totalMexicocount=MergeCount(totalMexicocount,thisMexicocount)
if line(2) == "Canada" :
        thisCanadacountt = AirportNameLetterFrequency(line(1))
        totalCanadacount=MergeCount(totalCanadacount,thisCanadacount)
                
#ahist, ehist, ihist, ohist, uhist, yhist==CounttoHistograms(thisUSAcount, totalUSAcount, totalEnglandcount, totalTurkeycount, totalFrancecount, totalGermanycount, totalMexicocount, totalCanadacount)


print totalUSAcount
print totalChinacount
print totalTurkeycount
print totalEnglandcount
print totalFrancecount
print totalGermanycount
print totalMexicocount
print totalCanadacount


#fig, ax = plt.subplots()
#index=np.arrange(n_groups)
#barwidth=0.25
#opacity = 0.8
#rects = plot.bar(index
