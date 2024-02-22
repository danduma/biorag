import torch
from llama_index.llms import Ollama

# Set device to CPU explicitly
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

from ragatouille import RAGPretrainedModel

rag = RAGPretrainedModel.from_index(".ragatouille/colbert/indexes/muscle_sample/")


def get_snippets(query, max_snippets=10):
    docs = rag.search(query, k=max_snippets)
    return [doc for doc in docs]


def concatenate_snippets(snippet_dicts):
    concatenated = {}
    for snippet in snippet_dicts:
        document_id = snippet['document_id']
        content = snippet['content']
        if document_id in concatenated:
            concatenated[document_id]['content'] += content
        else:
            concatenated[document_id] = {'content': content, 'document_id': document_id}
    return concatenated


def print_snippets(docs):
    print("Found", len(docs), "documents")
    for doc in docs:
        text = doc['content']
        print(text[:1000] + "..." if len(text) > 1000 else text)
        print('---------')


def llm_call(query, context):
    llm = Ollama(model="gemma:7b", base_url="http://127.0.0.1:11434")

    prompt = """
    The user asks the following query: "{}"

    You need to output only "YES" or "NO" as an answer to the following question: 

    "Does the provided document text provide support or counterargument for the user query?"
    
    You must output nothing but either the string "YES" or "NO". Your output should be at most 1 word in length. In no case should you output more words than a maximum of 1.

    The document text is the following:
    {}
    """
    text = prompt.format(query, context)
    res = llm.complete(text)
    print(res)
    return res


def answer_query(user_input):
    snippets = get_snippets(user_input)
    # print_snippets(docs)
    docs = concatenate_snippets(snippets)
    for doc_id, doc in docs.items():
        res = llm_call(user_input, doc['content'])
        doc['relevant'] = res

    print(docs)


def forever_user_input():
    snippets = get_snippets('')
    while 1:
        user_input = input("Enter your search query: ")
        answer_query(user_input)


def test_one():
    user_input = 'Did manipulation of a gene alter skeletal muscle vascularization or capillary density?'
    answer_query(user_input)


def main():
    test_one()


if __name__ == '__main__':
    main()
