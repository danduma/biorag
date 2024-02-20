from ragatouille import RAGPretrainedModel

rag = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/blog/")

while 1:
    user_input = input("Enter your search query: ")
    docs = rag.search(user_input)
    for doc in docs:
        print(doc.text[:1000] + "..." if len(doc.text) > 1000 else doc.text)
        print()
