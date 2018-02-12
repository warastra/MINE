import numpy as np
import matplotlib.pyplot as plt

def SplitToXYcvTrain(X,Y,CVchunk):
	data = np.array([X.values],[Y.values])
	np.random.shuffle(data)
	train = pd.DataFrame(data)
	Xcv = df.iloc[:df.shape[0]*CVchunk]
	train = df.iloc[df.shape[0]*CVchunk:]
	Ycv = Xcv[Xcv.columns[-1]]
	Ytrain = train[train.columns[-1]]
	Xcv.drop(Xcv.columns[-1])
	train.drop(train.columns[-1])
	return (Xcv,train,Ycv,Ytrain)

def predict(X,theta):
	return X*theta

def train(X,Y):
	return theta
	
def LearnCurvePlot(Ytrain, XTrain, Ycv, Xcv):
	nrows = Ytrain.shape[0]
	Jtrain = np.empty([11])
	Jcv = np.empty([11])
	for i in range(nrows/10,nrows+1,nrows/10):
		theta = train(XTrain[:(i+1)],Ytrain[:(i+1)])
		YpredTrain = predict(Xtrain,theta)
		Jtrain[10*i/m] = sum((YpredTrain-Ytrain[:(i+1)])**2)
		YcvPred = predict(Xcv,theta)
		Jcv[10*i/m] = sum ((YcvPred-Ycv)**2)
		
	m = [x for x in range(0,nrows+1,nrows/10)]
	plt.plot(m,Jtrain,'r--',m,Jcv,'bs')
	plt.show()
	
def getXYZ(filename):
	pos_data = []
	lat_data = []
	with open(filename,'r+') as f:
		for line in f.readlines():
				x = line.split()
				if x[0] == 'atom':
					pos_data.append([np.array(x[1:4],dtype = np.float),x[4]])
				elif x[0] == 'lattice_vector':
					lat_data.append(np.array(x[1:4],dtype = np.float))
	return(pos_data, np.array(lat_data))				

#amat = np.transpose(lattice vectors)
def get_shortest_distances(reduced_coords, amat):
    natom = len(reduced_coords)
    dists = np.zeros((natom, natom))
    Rij_min = np.zeros((natom, natom, 3))
    for i in range(natom):
        for j in range(i):
            rij = reduced_coords[i][0] - reduced_coords[j][0]
            d_min = np.inf
            R_min = np.zeros(3)
            for l in range(-1, 2):
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        r = rij + np.array([l, m, n])
                        R = np.matmul(amat, r)
                        d = length(R)
                        if d < d_min:
                            d_min = d
                            R_min = R
            dists[i, j] = d_min
            dists[j, i] = dists[i, j]
            Rij_min[i, j] = R_min
            Rij_min[j, i] = -Rij_min[i, j]
    return dists, Rij_min
	
def get_min_length(distances, A_atoms, B_atoms):
    A_B_length = np.inf
    for i in A_atoms:
        for j in B_atoms:
            d = distances[i, j]
            if d > 1e-8 and d < A_B_length:
                A_B_length = d
    
    return A_B_length	
				
	
				