import pandas as pd 
import os 
df = pd.read_csv('requirements_env_pytorch36.txt', delimiter = "\t")


for i in df.values:
    query="pip install "+str(i[0])
    os.system(query)