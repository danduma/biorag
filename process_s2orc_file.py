import gzip
import json
import os


def get_text(annotations, text, key):
    return "\n".join(get_values(annotations, text, key))


def get_values(annotations, text, key):
    spans = annotations.get(key, [])
    values = extract_text(text, spans)
    return values


def get_authors(annotations, text):
    authorfirstname = get_values(annotations, text, "authorfirstname")
    authorlastname = get_values(annotations, text, "authorlastname")
    authors_list = list(zip(authorfirstname, authorlastname))
    authors = [{"first": first, "last": last} for first, last in authors_list]
    return authors


# Function to extract text from a list of spans
def extract_text(text, spans):
    if isinstance(spans, str):
        spans = json.loads(spans)
    extracted_parts = []
    if spans:
        for span in spans:
            start, end = int(span["start"]), int(span["end"])
            extracted_parts.append(text[start:end])
    return extracted_parts


def process_record(record):
    text = record["content"]["text"]
    annotations = record["content"]["annotations"]

    return {
        "title": get_text(annotations, text, "title"),
        "abstract": get_text(annotations, text, "abstract"),
        "text": text,
        "authors": get_authors(annotations, text),
        "author_text": get_values(annotations, text, "author"),
        "venue": get_values(annotations, text, "venue"),
        "externalids": record["externalids"],
        "source": record["content"]["source"],
        "corpusid": record["corpusid"],
    }


def process_file(file_name: str, output_directory=None, max_papers_per_file=10000, output_file_prefix=''):
    # Read the JSON data
    outfile_counter = 0
    processed_counter = 0

    if not output_directory:
        output_directory = os.path.dirname(file_name)

    with gzip.open(file_name, 'rt') as f:
        outfile = gzip.open(os.path.join(output_directory, f'{output_file_prefix}{outfile_counter}.gz'), 'wt')
        for line in f:
            record = json.loads(line)
            processed = process_record(record)
            outfile.write(json.dumps(processed))
            outfile.write('\n')
            processed_counter += 1
            if processed_counter % max_papers_per_file == 0:
                outfile.close()
                outfile_counter += 1
                outfile = gzip.open(os.path.join(output_directory, f'{output_file_prefix}{outfile_counter}.gz'),
                                    'wt')


def main():
    process_file('/Users/masterman/NLP/20240301_122116_00026_cdwgd_017bf577-41e1-4d66-8133-428bb9adde4a.gz',
                 output_file_prefix='s2orc_')


if __name__ == '__main__':
    main()
