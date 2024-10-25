import re
import os
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TASK2_ROOT = "Task2"

def extract_sentence_content(xml_string):
    pattern = re.compile(r'<sentence id="([^"]+)">(.*?)</sentence>')

    # Find all matches in the XML string
    matches = pattern.findall(xml_string)

    return matches
def clean_text(text):
    # Remove special characters and digits
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text
def get_sentences_data(file_path):
    with open(TASK2_ROOT+"/"+file_path, 'r', encoding='latin-1') as file:
        sentences_data = extract_sentence_content(file.read())
    return sentences_data

file_list_paths = os.listdir(TASK2_ROOT)
file_list_paths = sorted(file_list_paths)
file_list_lens = []
file_list_lens_acc = [0]

whole_doc = []

for path in file_list_paths:
    sentences_data = get_sentences_data(path)
    file_list_lens.append(len(sentences_data))
    file_list_lens_acc.append( file_list_lens_acc[-1] + len(sentences_data))
    whole_doc.append(sentences_data)

print(f"Training corpus: {len(whole_doc)}")

whole_doc_sentence_data = [sent for doc in whole_doc for sent in doc]
whole_doc_content = [sent[1] for sent in whole_doc_sentence_data]

tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(whole_doc_content)

def get_file_indices(file_path):
    index_of_file = file_list_paths.index(file_path)
    x1 = file_list_lens_acc[index_of_file]
    x2 = file_list_lens_acc[index_of_file+1]
    return x1, x2

def extractive_summarization(file_path):
    # Extract sentences and clean text
    x1, x2 = get_file_indices(file_path)
    sentence_similarity_matrix = cosine_similarity(tfidf_matrix[x1:x2], tfidf_matrix[x1:x2])
    # Get the most informative sentences
    summary_indices = sentence_similarity_matrix.sum(axis=1).argsort()[-5:][::-1]
    return summary_indices


def summarizer(file_path):
    summary = ""
    sentences_data = get_sentences_data(file_path)
    summary_indices = extractive_summarization(file_path)
    for i, index in enumerate(summary_indices):
        summary += f"Sentence {i+1}: " + sentences_data[index][1] + "\n"
    print(f"Summary for {file_path}:")
    print(summary)
    return summary

summary = summarizer("06_3.xml")
summary = summarizer("07_963.xml")
summary = summarizer("07_1872.xml")
summary = summarizer("09_554.xml")
summary = summarizer("09_1598.xml")





