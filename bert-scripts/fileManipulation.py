import pandas as pd

def fimport(filename,sep=',',encoding='utf-16',skiprows=0, header=0):
	frame = pd.DataFrame()
	frame = pd.read_csv(filename,sep=sep, skiprows=skiprows, header=header, error_bad_lines=False)
	return frame