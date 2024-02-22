import torch
import os
from config.definitions import ROOT_DIR

# Set device to CPU explicitly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

from ragatouille import RAGPretrainedModel
import re
from llama_index import SimpleDirectoryReader

_tags_re = re.compile(r'<[^>]+>')


def strip_html_tags(text):
    return _tags_re.sub('', text)


def main(input_docs_dir):
    # db = sqlite_utils.Database("simonwillisonblog.db")
    rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    # entries = list(db["blog_entry"].rows)

    docs_path = os.path.join(ROOT_DIR, input_docs_dir)
    reader = SimpleDirectoryReader(input_dir=docs_path)

    all_docs = reader.load_data()

    entry_texts = []
    for doc in all_docs:
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
        index_name="muscle_sample",
        max_document_length=180,
        split_documents=True
    )


if __name__ == "__main__":
    main('data/muscle_papers_sample/')
