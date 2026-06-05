import faiss
import numpy as np

dimension = 384

index = faiss.IndexFlatL2(dimension)

vectors = np.random.random((10, dimension)).astype("float32")

index.add(vectors)

print(f"Nombre de vecteurs dans FAISS : {index.ntotal}")