import os
import pandas as pd
import shutil
from config.definitions import ROOT_DIR


def load_papers(directory_path):
    filenames = [x for x in os.listdir(directory_path) if x.endswith('.txt')]
    papers = pd.DataFrame({'filename': filenames})
    papers['pmcid'] = papers['filename'].apply(lambda x: x.split('.')[0])
    papers
    return papers


def load_golden_papers(golden_papers_path):
    cols = ['pmcid']
    golden_papers = pd.read_csv(golden_papers_path, usecols=cols, sep='\t', encoding='latin-1')
    golden_papers = golden_papers.dropna().drop_duplicates()
    return golden_papers


def sample_papers(papers_path, golden_papers_path, n_samples=10000):
    papers = load_papers(papers_path)
    golden_papers = load_golden_papers(golden_papers_path)
    sample_golden_papers = papers.merge(golden_papers, on='pmcid')
    sample_non_golden_papers = papers[~papers['pmcid'].isin(golden_papers['pmcid'])].sample(n_samples)
    papers_sample = pd.concat([sample_golden_papers, sample_non_golden_papers])
    return papers_sample


def save_sample_files(papers_sample, sample_dir, papers_path):
    filenames = papers_sample['filename'].values
    for filename in filenames:
        src_path = os.path.join(papers_path, filename)
        dst_path = os.path.join(sample_dir, filename)
        shutil.copyfile(src_path, dst_path)


def main():
    papers_path = os.path.join(ROOT_DIR, 'data/raw_papers/')
    golden_papers_path = os.path.join(ROOT_DIR, 'data/golden_papers_manual_review.tsv')
    papers_sample = sample_papers(papers_path, golden_papers_path)
    sample_dir = os.path.join(ROOT_DIR, 'data/muscle_papers_sample/')
    save_sample_files(papers_sample, sample_dir)


if __name__ == "__main__":
    main()
