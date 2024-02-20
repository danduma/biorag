from ragatouille import RAGPretrainedModel
import re
from llama_index import SimpleDirectoryReader

_tags_re = re.compile(r'<[^>]+>')


def strip_html_tags(text):
    return _tags_re.sub('', text)


def main():
    # db = sqlite_utils.Database("simonwillisonblog.db")
    rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    # entries = list(db["blog_entry"].rows)

    reader = SimpleDirectoryReader(input_dir="/Users/masterman/NLP/pmcoas/PMC001xxxxxx/")

    all_docs = reader.load_data()

    entry_texts = []
    for docs in reader.iter_data():
        for doc in docs:
            entry_texts.append(doc.text)

    print("len of entry_texts is", len(entry_texts))
    entry_ids = [str(doc.doc_id) for doc in all_docs]
    entry_metadatas = [
        {"slug": doc.doc_id} for doc in all_docs
    ]

    rag.index(
        collection=entry_texts,
        document_ids=entry_ids,
        document_metadatas=entry_metadatas,
        index_name="blog",
        max_document_length=180,
        split_documents=True
    )


if __name__ == "__main__":
    main()
