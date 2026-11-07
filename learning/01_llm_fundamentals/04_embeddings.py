
import ollama

sentences = [
    "Kubernetes pod keeps restarting",
    "My container is continuously crashing",
    "I enjoy watching anime"
]

for sentence in sentences:

    response = ollama.embed(
        model="all-minilm",
        input=sentence
    )

    vector = response["embeddings"][0]

    print("\nSentence:", sentence)
    print("Embedding dimensions:", len(vector))
    print("First five values:", vector[:5])
