import pandas as pd
import numpy as np
import ast
from numpy import dot
from numpy.linalg import norm
from sentence_trans import embeddings
def fun(x):
    actual_list = ast.literal_eval(x)
    return np.array(actual_list)

def find_sim(x):
    cos_sim = dot(x, em2)/(norm(x)*norm(em2))
    return cos_sim

user_query = "How much money i have spent on online shopping"
em2 = embeddings(user_query)
df = pd.read_csv('data/data_embed.csv')
df.fillna(0,inplace=True)
df['embeddings'] = df['embeddings'].apply(fun)
df['sim'] = df['embeddings'].apply(find_sim)
df.sort_values(by='sim',ascending=False,inplace=True)
df = df[df['sim']>0.2]
print(df)
print("Response: ",sum(df['paid out'].tolist()))