from sentence_transformers import SentenceTransformer
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('data/data.csv')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def embed(desc):
    embeddings = model.encode(desc)
    return list(embeddings)

df['embeddings'] = df['description'].apply(embed)
df.to_csv('data/data_embed.csv',index=False)

