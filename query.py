import torch

# Set device to CPU explicitly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

from ragatouille import RAGPretrainedModel

rag = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/blog/")

while 1:
    user_input = input("Enter your search query: ")
    docs = rag.search(user_input)
    for doc in docs:
        text = doc['content']
        print(text[:1000] + "..." if len(text) > 1000 else text)
        print()
