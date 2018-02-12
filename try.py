import pandas as pd
import numpy as np
import MLhelp as mh
import helper2 as h2
import math

head = r"D:\Advance\Datasets\Kaggle\Transparent Conductors\train"
files = h2.list_file(head,2400)
df = h2.sDist(head+"\\"+'1'+r"\geometry.xyz")
print df

df = pd.DataFrame(np.transpose(df.as_matrix(columns=['distance (A)'])))
for i in files:
	dfnext = h2.sDist(i)
	df = pd.concat([df,pd.DataFrame(np.transpose(dfnext.as_matrix(columns=['distance (A)'])))])
df.to_csv(head+r"\r3.csv")
