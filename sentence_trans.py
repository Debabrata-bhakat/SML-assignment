from sentence_transformers import SentenceTransformer
sentences = "This is an example sentence", "Each sentence is converted"


def embeddings(x):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(x)

