import numpy as np
import pandas as pd
import MLhelp as mh
import math

def list_file(head_addr,num):
	address = []
	for i in range(2,num+1):
		address.append(head_addr+"\\"+str(i)+r"\geometry.xyz")
	return address
	
#input: df(getXYZ)
def MINdist(xtaldf,atomA,atomB):
	aLoc = xtaldf[xtaldf[1]==atomA]
	aLoc = np.array(aLoc[0])
	
	
	bLoc = xtaldf[xtaldf[1]==atomB]
	bLoc = np.array(bLoc[0])
	
	#distance = 99999
	temp_min1 = 10000
	temp_min2 = 10000
	temp_min3 = 10000
	for i in range(0,aLoc.shape[0]):
		for j in range(0,bLoc.shape[0]):
			distance = math.sqrt((aLoc[i][0]-bLoc[j][0])**2+(aLoc[i][1]-bLoc[j][1])**2+(aLoc[i][2]-bLoc[j][2])**2)
			if  distance<1e-2:
				continue
			elif temp_min1>distance:
				temp_min3 = temp_min2
				temp_min2 = temp_min1
				temp_min1 = distance
	if temp_min3>100 or temp_min3 is None:
		temp_min3 = 0
	finDist = temp_min3
	id = atomA+" and "+atomB
	return finDist,id
	
def sDist(addr):
	xtal,lat = mh.getXYZ(addr)
	xtaldf = pd.DataFrame(np.array(xtal))
	
	atomDist = []
	for i in ['Ga','Al','In','O']:
		dist, name = MINdist(xtaldf,'Ga',i)
		atomDist.append([np.array(name,dtype=str),dist])
	for i in ['Al','In','O']:
		dist,name = MINdist(xtaldf,'Al',i)
		atomDist.append([np.array(name,dtype=str),dist])

	for i in ['In','O']:
		dist,name = MINdist(xtaldf,'In',i)
		atomDist.append([np.array(name,dtype=str),dist])
	
	dist,name = MINdist(xtaldf,'O','O')
	atomDist.append([np.array(name,dtype=str),dist])
	
	atomDist = pd.DataFrame(atomDist,columns=['atoms','distance (A)'])
	return atomDist