import torch
import pandas as pd

# Set device to CPU explicitly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

from ragatouille import RAGPretrainedModel


def query_index(query, index_path):
    rag = RAGPretrainedModel.from_index(index_path)
    docs = rag.search(query,k=100)
    print ("Found", len(docs), "documents")
    docs_df = pd.DataFrame(docs)
    docs_df['file_name'] = [x['document_metadata']['file_name'] for x in docs]
    docs_df['pmcid'] = docs_df['file_name'].apply(lambda x: x[:-4])
    cols = ['document_id','pmcid','passage_id','rank','score','content']
    docs_df = docs_df.sort_values('rank')
    docs_df = docs_df[cols]
    return docs_df

def doc_ranking(ranked_snippets):
    ranked_docs = ranked_snippets.groupby('pmcid').agg(
        rank = ('rank',min),
        snippets = ('content',list),
        snippet_scores = ('score',list)
    ).reset_index()
    return ranked_docs.sort_values('rank')

def load_golden_papers(golden_papers_path):
    cols = ['pmcid','in_pmc_oas']
    golden_papers = pd.read_csv(golden_papers_path, usecols = cols)
    golden_papers = golden_papers[golden_papers['in_pmc_oas']]
    golden_papers = golden_papers[['pmcid']]
    return golden_papers

def load_golden_papers_reviews(golden_papers_reviews_path):
    golden_papers_reviews = pd.read_csv(golden_papers_reviews_path, sep='\t')
    return golden_papers_reviews

def golden_paper_reviews_ranking(golden_papers_reviews, ranked_papers_reviews):
    has_yes = lambda x: 'Yes' if 'Yes' in list(x) else 'No'
    ranked_golden_papers_reviews = golden_papers_reviews.groupby(['pmid','pmcid','question'],sort=False).agg(answer=('answer',has_yes))
    ranked_golden_papers_reviews = ranked_golden_papers_reviews.reset_index()
    ranked_golden_papers_reviews = ranked_golden_papers_reviews.merge(ranked_papers_reviews, on=['pmcid','question'])
    return ranked_golden_papers_reviews

def evaluate_search(ranked_papers,golden_papers_reviews):

    questions = set(golden_papers_reviews['question'])
    queries = [question.replace('Did ','').replace('?','') for question in questions]

    index_path = ".ragatouille/colbert/indexes/muscle_sample_pmc/"
    cols = ['pmcid','question','rank']
    ranked_papers_reviews = pd.DataFrame(columns=cols)
    for query in queries:
        ranked_snippets = query_index(query, index_path)
        ranked_papers = doc_ranking(ranked_snippets).sort_values('rank')
        ranked_papers['question'] = 'Did {}?'.format(query)
        ranked_papers_reviews = pd.concat([ranked_papers_reviews,ranked_papers[cols]])
    
    ranked_papers_reviews = golden_paper_reviews_ranking(golden_papers_reviews, ranked_papers_reviews)
    return ranked_papers_reviews


def main():
    #golden_papers_path = 'data/muscle_golden_papers.csv'
    #golden_papers = load_golden_papers(golden_papers_path)

    golden_papers_reviews_path = 'data/muscle_pmc_golden_papers_manual_reviews.tsv'
    golden_papers_reviews = load_golden_papers_reviews(golden_papers_reviews_path)

    ranked_papers_reviews = evaluate_search(golden_papers_reviews,golden_papers_reviews)
    pass


if __name__ == "__main__":
    main()