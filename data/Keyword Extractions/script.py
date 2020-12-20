import pandas as pd
import numpy as np
import os
import glob

def fimport(filename,sep=',',encoding='utf-16',skiprows=1, header=0):
	frame = pd.DataFrame()
	frame = pd.read_csv(filename,sep=sep, skiprows=skiprows, header=header, encoding=encoding, error_bad_lines=False)
	return frame
	
filename = 'test.csv'
df = fimport(filename)

print(len(df))
print(df.head())

print(df[0][2])

