import torch
import pandas as pd

# Set device to CPU explicitly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

from ragatouille import RAGPretrainedModel


def query_index(query, index_path):
    rag = RAGPretrainedModel.from_index(index_path)
    docs = rag.search(query)
    print ("Found", len(docs), "documents")
    cols = ['document_id','passage_id','rank','score','content']
    docs_df = pd.DataFrame(docs)[cols].sort_values('rank')
    return docs_df

def main():
    query = 'Manipulation gene alters skeletal muscle or skeletal myocyte oxidative capacity'
    index_path = ".ragatouille/colbert/indexes/muscle_sample/"
    query_index(query, index_path)

if __name__ == "__main__":
    main()