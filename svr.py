import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn import preprocessing as prep
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor

doc = r"D:\Advance\Datasets\Kaggle\Transparent Conductors\trainModif.csv"
test = r"D:\Advance\Datasets\Kaggle\Transparent Conductors\test.csv"

DRed = TruncatedSVD(n_components=48)

def normalize(x):
	for j in range(x.shape[1]):
		x[1:,j] = (x[1:,j]-x[1:,j].mean())/np.std(x[1:,j])
		return x
	
def nomad_prep(train,skip):
	cat = pd.read_csv(train,usecols=[0,1,2],skiprows=skip)
	cat1 = pd.get_dummies(cat['spacegroup'])
	cat2 = pd.get_dummies(cat['number_of_total_atoms'])
	mastercat = pd.concat([cat1,cat2],axis=1)
	numer =  pd.read_csv(train,usecols=[x for x in range(3,43)],skiprows=skip)
	numer = pd.DataFrame(normalize(numer.values))
	#numer = pd.DataFrame(DRed.fit_transform(numer))
	#z = pd.DataFrame([[numer['Al']/numer["Ga"]],[numer["Al"]/numer['In']],[numer['Ga']/numer['In']]])
	#v = pd.DataFrame([[numer['x']/numer['y']],[numer['x']/numer['y']],[numer['y']/numer['z']]])
	#angle = pd.DataFrame([[numer['alpha']/numer['beta']],[numer['alpha']/numer['gamma']],[numer['gamma']/numer['beta']]])
	#X = pd.concat([mastercat, numer,z,v,angle],axis=1)
	X = pd.concat([mastercat, numer],axis=1)
	return X
	
def AddOnes(X):
	X = pd.concat([X,pd.DataFrame(np.ones((X.shape[0],1)))],axis = 1)
	return X
	
HformDF = pd.read_csv(doc,usecols=[43])
bandgapDF = pd.read_csv(doc,usecols=[44])
Hform = HformDF.as_matrix()
bandgap = bandgapDF.as_matrix()
#split train and CV
skiplist1 = [x for x in range(2001,2401)] 	
Xtrain = nomad_prep(doc,skiplist1)
Xtrain = Xtrain.fillna(0.0)
print Xtrain.shape
polyfeat = prep.PolynomialFeatures(degree = 4, include_bias=False)
Xtrain = pd.DataFrame(polyfeat.fit_transform(Xtrain,y=Hform[:2000,:]))
Xtrain = pd.DataFrame(DRed.fit_transform(Xtrain))
Xtrain = AddOnes(Xtrain)
#print Xtrain.head()
#print '\n'

skiplist2 = [x for x in range(1,2001)]
XCV = nomad_prep(doc,skiplist2)
XCV = XCV.fillna(0.0)
print XCV.shape
XCV = pd.DataFrame(polyfeat.transform(XCV))
XCV = pd.DataFrame(DRed.transform(XCV))
XCV = AddOnes(XCV)

Xtest = nomad_prep(test,0)
Xtest = Xtest.fillna(0.0)
print Xtest.shape
Xtest = pd.DataFrame(polyfeat.transform(Xtest))
Xtest = pd.DataFrame(DRed.transform(Xtest))
Xtest = AddOnes(Xtest)

#ML technique

#BGalgain = MLPRegressor(activation='identity',hidden_layer_sizes=(100,40,5,5),learning_rate_init=0.002,alpha=0.1)
#HFalgain = MLPRegressor(activation='identity',hidden_layer_sizes=(100,40,5,5),learning_rate_init=0.005)
#BGalgain = svm.SVR(kernel='poly', degree = 5, coef0 = 1,gamma= 0.01, C =1e3)
#HFalgain = svm.SVR(kernel='rbf', C=1e3, gamma = 0.1)
BGalgain = RandomForestRegressor(max_depth=15,n_estimators=40,max_features=25)
HFalgain = RandomForestRegressor(max_depth=20,n_estimators=40,max_features=30)

#cross validation
bgTAcc = np.empty([11])
bgCVAcc = np.empty([11])
hfTAcc = np.empty([11])
hfAcc = np.empty([11])

for m in range(200,2001,200):
	BGalgain.fit(Xtrain.iloc[0:m].values,bandgap[0:m,:].ravel())
	bgHX = BGalgain.predict(Xtrain.iloc[0:m])
	bgTAcc[m/200] = sum((bandgap[0:m,:].ravel()-bgHX)**2)/m
	bgPred = BGalgain.predict(XCV)
	bgCVAcc[m/200] = sum((bandgap[2000:2400,:].ravel()-bgPred)**2)/400
	
	HFalgain.fit(Xtrain.iloc[0:m].values,Hform[0:m,:].ravel())
	hfHX = HFalgain.predict(Xtrain.iloc[0:m])
	hfTAcc[m/200] = sum((Hform[0:m,:].ravel()-hfHX)**2)/m
	hfPred = HFalgain.predict(XCV)
	hfAcc[m/200] = sum((Hform[2000:2400,:].ravel()-hfPred)**2)/400

"""
#optimize parameter
for m in range(5,41,5):
	BGalgain = RandomForestRegressor(max_depth=m,n_estimators=500,max_features=25)
	BGalgain.fit(Xtrain.values,bandgap[0:2000,:].ravel())
	bgHX = BGalgain.predict(Xtrain)
	bgTAcc[m/5] = sum((bandgap[0:2000,:].ravel()-bgHX)**2)/2000
	bgPred = BGalgain.predict(XCV)
	bgCVAcc[m/5] = sum((bandgap[2000:2400,:].ravel()-bgPred)**2)/400
	
	HFalgain = RandomForestRegressor(max_depth=m,n_estimators=500,max_features=30)
	HFalgain.fit(Xtrain.values,Hform[0:2000,:].ravel())
	hfHX = HFalgain.predict(Xtrain)
	hfTAcc[m/5] = sum((Hform[0:2000,:].ravel()-hfHX)**2)/2000
	hfPred = HFalgain.predict(XCV)
	hfAcc[m/5] = sum((Hform[2000:2400,:].ravel()-hfPred)**2)/400
"""

print "bandgap train:"
print bgTAcc
print "bandgap CV:"
print bgCVAcc

print "energy of formation train:"
print hfTAcc
print "energy of formation:"
print hfAcc

m = [x for x in range(0,2001,200)]
#h = plt.figure(1)
#plt.plot(m,hfTAcc,'r--',m,hfAcc,'rs')
#h.show()
b = plt.figure(2)
plt.plot(m,bgTAcc,'b--',m,bgCVAcc,'bs')
plt.show()
raw_input()

#predicting test set
#XX = pd.concat([Xtrain,XCV],ignore_index=True)
#BGalgain.fit(Xtrain,bandgap[0:2400,:].ravel())
#HFalgain.fit(Xtrain,Hform[0:2400,:].ravel())
bgtest = BGalgain.predict(Xtest)
hftest = HFalgain.predict(Xtest)
z = pd.DataFrame(hftest,bgtest)
z.to_csv("submit2.csv")
