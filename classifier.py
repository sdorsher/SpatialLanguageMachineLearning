#!/usr/bin/python
import os, sys

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import cauchy
import numpy as np
from io import StringIO
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression

file = open("airports-extended.dat.txt","r")

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



def LatLongClassMaker(coor):
    lat,long=coor
    if lat<30:
        if long>-90:
            return 1 #florida
        elif long < -120:
            return 2 #hawaii
        else:
            return 3 #texas
    elif lat >50:
        return 4 #alaska
    elif long> -80 and lat < 40:
        return 5 #south east
    elif long> -80 and lat >40:
        return 6 #new england
    elif long< -80:
        return int((50-lat)/20.*5)*int((125+long)/45.*9)+7
    else:
        return 0
    #worst mapping ever-- ignores new england and city diversity
    return 0
ratio=[]
coords=[]


for line in file:
    line=re.split(',',line);

    if line[3] == "\"United States\"":
        thisratioV, thisratioS, thisratioO=VowelToConsRatio(line[1]);
        ratio.append([thisratioV,thisratioS,thisratioO])
        coords.append([float(line[6]),float(line[7])])


ratioarray=np.empty([len(ratio),3])
coordsarray=np.empty([len(coords),2])
regionarray=np.empty([len(coords)])

for i, rat in enumerate(ratio):
    ratioarray[i,:]=rat
for i,coor in enumerate(coords):
    coordsarray[i,:]=coor
#    outputcoor=LatLongClassMaker(coor)
#    regionarray[i]=outputcoor


xtrain,xtest,ytrain,ytest=train_test_split(ratioarray,coordsarray,test_size=0.33,shuffle=True, random_state=42)
testphrases=["Ufda!", "I just dont know about that", "Howdy, pardner", "Where are we?", "I thank whatever gods may be for my unconquerable soul.", "I welcome our new robot overlords", "l33t h4k3rs", "I know you I walked with you once upon a dream", "These stories dont mean anything if you have no one to tell them to", "Baby you have the sort of hands that rip me apart","Multi-messenger astronomy", "Rainbow flag", "I prefer They or He?", "numerical relativity", "LIGO", "scalar field", "Osculating Orbits", "Monte-Carlo Simulation", "Data Analysis", "Data Science", "Parallelization", "Paralyzation", "Partial disability", "Non-epileptic seizures", "Wednesday Lunch", "Tuesday Lunch", "Thursday Lunch", "Guild Wars", "Elvenar", "Good Apple"]

colleaguephrases=["general relativity", "black hole", "loop quantum gravity",  "quantization", "space-time", "Hamiltonian constraint", "Ashtekar", "LiSA", "LIGO", "group", "white hole", "scalar field", "numerical relativity", "cosmology", "diffeomorphism", "continuum limit", "David Berger", "Reisner-Nordstrom", "black hole spacetime", "initial data", "interpretation of quantum mechanics"]

xphrases=[]
for phrase in testphrases:
    print(phrase)
    vtc, stc, otc = VowelToConsRatio(phrase)
    xphrases.append([vtc,stc,otc])
xcolleague=[]
for phrase in colleaguephrases:
    print(phrase)
    vtc, stc, otc = VowelToConsRatio(phrase)
    xcolleague.append([vtc,stc,otc])



scaler = StandardScaler()
xtrain2=scaler.fit_transform(xtrain)
xtest2=scaler.fit_transform(xtest)
xphrases2=scaler.fit_transform(xphrases)
xcolleague2=scaler.fit_transform(xcolleague)


#reimplementing OneHotEncoder, in order to map numeric data to geographic output classes for ytrain
#maxval=int(max(ytrain))
#print(maxval)
#ytrainmatrix=np.zeros([len(ytrain),maxval+1])
#ytestmatrix=np.zeros([len(ytest),maxval+1])
#for i in np.arange(maxval):
#    ytrainmatraix[i,int(ytrain)]=1
#    ytestmatrix[i,int(ytest)]=1


#change to poisson input data instead of ratio input data? I think the power law tail is what is being missed and what is causing the validation set to shrink. 

kpca=KernelPCA(n_components=9, random_state=42, kernel="poly", degree=3, tol=1e-3, max_iter=None, remove_zero_eig=False, fit_inverse_transform=True)
X_kpca_train=kpca.fit_transform(xtrain2)
X_kpca_test=kpca.fit_transform(xtest2)
X_kpca_phrases=kpca.fit_transform(xphrases2)
X_kpca_colleague=kpca.fit_transform(xcolleague2)

print(X_kpca_phrases[0])
print(X_kpca_train[0])
print(X_kpca_colleague[0])

print(np.shape(X_kpca_train))
print(np.shape(ytrain))
lr3to2_lat=LinearRegression()
lr3to2_long=LinearRegression()
lr3to2_lat.fit(X_kpca_train,ytrain[:,0])
lr3to2_long.fit(X_kpca_train,ytrain[:,1])



latparams=lr3to2_lat.get_params()
longparams=lr3to2_long.get_params()
latprediction=lr3to2_lat.predict(X_kpca_train)
longprediction=lr3to2_long.predict(X_kpca_train)
phraselatpred=lr3to2_lat.predict(X_kpca_phrases)
phraselongpred=lr3to2_long.predict(X_kpca_phrases)
phraselatcolleague=lr3to2_lat.predict(X_kpca_colleague)
phraselongcolleague=lr3to2_long.predict(X_kpca_colleague)


print(testphrases)
print(phraselatpred)
print(phraselongpred)
print(ytrain[:,0].mean(), ytrain[:,1].mean())

plt.figure()
plt.title("Verification data for latitudes and longitudes in the US")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.scatter(ytrain[:,1], ytrain[:,0], c='c', label="Training set")
plt.scatter(longprediction,latprediction,c='r', label="Validation set")
plt.scatter(phraselongpred,phraselatpred,c='b',label="My test phrases, I live in Minnesota right now")
plt.scatter(phraselongcolleague, phraselatcolleague, c='y', label="Colleagues test phrases, he lives in Louisiana right now")
#plt.scatter(phraselongpred[0],phraselatpred[1],c='b',label=testphrases[0])
#plt.scatter(phraselongpred[1],phraselatpred[1],c='m',label=testphrases[1])
#plt.scatter(phraselongpred[2],phraselatpred[1],c='c',label=testphrases[2])
#plt.scatter(phraselongpred[3],phraselatpred[1],c='g',label=testphrases[3])
#plt.scatter(phraselongpred[4],phraselatpred[1],c='y',label=testphrases[4])
#plt.scatter(phraselongpred[5],phraselatpred[1],c='y',label=testphrases[5])
#plt.scatter(phraselongpred[6],phraselatpred[1],c='y',label=testphrases[6])
#plt.scatter(phraselongpred[7],phraselatpred[1],c='y',label=testphrases[7])
#plt.scatter(phraselongpred[8],phraselatpred[1],c='y',label=testphrases[8])
#plt.scatter(phraselongpred[9],phraselatpred[1],c='y',label=testphrases[9])
#plt.scatter(phraselongpred[10],phraselatpred[1],c='y',label=testphrases[10])
#plt.scatter(phraselongpred[11],phraselatpred[1],c='y',label=testphrases[11])
plt.legend(loc="lower right")
plt.show()



#clf=OneVsRestClassifier(MultinomialNB())
#clf=MLPClassifier(hidden_layer_sizes=(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#clf.fit(xtrain2,ytrain)
#predictions=clf.predict(xtrain2)
#distance=clf.decision_function(xtrain2)

#accuracy=accuracy_score(ytest,prediction)
#print(accuracy)

#fig=plt.figure()
#ax=fig.gca()


#ax.scatter(xtrain2[:,1],xtrain2[:,0],c=predictions,cmap=cm.plasma_r)

#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.title('Map of language classes for US using extended training set')
#plt.show()



