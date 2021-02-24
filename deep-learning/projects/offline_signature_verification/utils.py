import numpy as np

def CosineSimilarity(u, v):
    # compute the dot product between the word
    dot = u @ v
    # Compute the L2 norm of u
    norm_u  = np.linalg.norm(u)
    # Compute the L2 norm of v
    norm_v = np.linalg.norm(v)
    # Compute the cosine similarity by formula
    cosine_similarity = dot / (norm_u * norm_v)
    
    return cosine_similarity
    