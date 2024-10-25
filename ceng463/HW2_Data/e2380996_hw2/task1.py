from pathlib import Path

import spacy
from spacy.tokens import Doc, DocBin
from spacy.training.example import Example
import conllu
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.lang.tr import Turkish
from spacy_conll import ConllParser, init_parser
from spacy import Language
from spacy.scorer import Scorer

# Read CoNLL-U data
train_data_path = "Task1/tr_imst-ud-train.conllu"
dev_data_path = "Task1/tr_imst-ud-dev.conllu"
test_data_path = "Task1/tr_imst-ud-test.conllu"

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        annotations = file.read()
    sentences = conllu.parse(annotations)
    return sentences

train_sentences = load_data(train_data_path)
dev_sentences = load_data(dev_data_path)
test_sentences = load_data(test_data_path)

import spacy
from spacy.training.example import Example
import conllu

# Assuming you already have your spaCy model loaded
nlp = spacy.load("output/model-best")
parser = nlp.get_pipe('parser')

print("parser")
print(parser)

def print_res(res):
    print(f"Sentence: {res.text}")
    for token in res:
        print(f"token: {token.text} - dep: {token.dep_} - head: {token.head.text} - head_dep: {token.head.dep_}")
    print("-------------------")

my_sent = [
    "Güzel bir günbatımında deniz kıyısında yürüyordum.",

    "Ankara'da kışın genellikle kar yağar.",

    "Kardeşimin evleneceğini söylediği günün ertesi günü gördüğüm kırmızı "
    "arabayı tekrar görmem sadece on dakika sürdü.",

    "Geçen gün seni gördüğümde gözlerindeki yaşı anlamış gibi alev atarken hiç de utanmam yoktu."
]

results = [ nlp(sent) for sent in my_sent ]

for res in results:
    print_res(res)






