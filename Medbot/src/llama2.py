from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the model
model = SentenceTransformer("thenlper/gte-large")

# Sentences to encode
sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]

# Encode the sentences to get embeddings
embeddings = model.encode(sentences)

# Compute cosine similarity between embeddings
similarities = cosine_similarity(embeddings)

# Print similarity matrix
print(similarities.shape)  # Should be (4, 4)
print(similarities)

