import nltk
import sklearn.naive_bayes

from sklearn.preprocessing import StandardScaler
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score, confusion_matrix
import string
import pickle  # this is for saving and loading your trained classifiers.
from nltk.corpus import stopwords
import re
import numpy as np

CATEGORY_TYPES = ["horror", "mystery", "philosophy", "religion", "romance", "science-fiction", "science", "sports"]
PREPROCESS_CUSTOM_FILTER_LST = ["'s", "n't", "--", "...", "·", "one", "world", "i", "new", "life"]

DEV = "dev"
TEST = "test"
TRAIN = "train"

DEV_PATH = f"data/{DEV}/"
TEST_PATH = f"data/{TEST}/"
TRAIN_PATH = f"data/{TRAIN}/"

DEV_EXT = f"_{DEV}.txt"
TEST_EXT = f"_{TEST}.txt"
TRAIN_EXT = f"_{TRAIN}.txt"

CORPUS_TYPE = "corpus_type"
CATEGORY_KEY = "category"
CATEGORY_LST_KEY = f"{CATEGORY_KEY}_lst"
FILM_KEY = "film"
FILM_LST_KEY = f"{FILM_KEY}_lst"
DESC_KEY = "desc"


### ***** YOU MAY A~DD NEW FUNCTIONS, MODIFY OR DELETE EXISTING ONES. YOU MAY EVEN TOTALLY DISCARD THIS TEMPLATE AND CODE YOUR OWN SOLUTION FROM SCRATCH. ***** ###


class Module:
    X_train = []
    y_train = []
    train_samples = []
    train_cl_samples = []

    X_test = []
    y_test = []
    test_samples = []
    test_cl_samples = []

    X_dev = []
    y_dev = []
    dev_samples = []
    dev_cl_samples = []

    def __init__(self):
        self.X_train, self.y_train, self.train_samples, self.train_cl_samples = self.create_megadoc(TRAIN)
        self.X_dev, self.y_dev, self.dev_samples, self.dev_cl_samples = self.create_megadoc(DEV)
        self.X_test, self.y_test, self.test_samples, self.test_cl_samples = self.create_megadoc(TEST)
        for freq, class_name in self.train_cl_samples:
            print(f"{class_name}: {freq.N()}")

    def preprocessData(self, data):
        data = self.remove_numbers_regex(data)
        data = data.lower()
        # Tokenization
        words = nltk.tokenize.word_tokenize(data)
        # Filtering with stop words
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words
                          if (
                                  word not in stop_words
                                  and
                                  word not in string.punctuation
                                  and
                                  word not in PREPROCESS_CUSTOM_FILTER_LST
                          )
                          ]
        freqDist = nltk.FreqDist(filtered_words)

        return freqDist

    def getDataForClass(self, filepath,
                        class_name):  # filename is of type string. example call: preprocess("philosophy_test.txt")

        print(f"classname: {class_name}")
        dataWclass = []
        with (open(filepath, 'r') as file):
            lines = file.readlines()
            for index in range(len(lines) // 2):
                name = lines[index * 2]
                desc = lines[index * 2 + 1]
                data = self.preprocessData(desc)
                dataWclass.append((data, class_name))
        return dataWclass

    def remove_numbers_regex(self, input_str):
        return re.sub(r'\d+', '', input_str)
    def r(self, input_str):
        return re.sub(r'[^\w\s]','',input_str)
    def create_megadoc(self, type):
        print(f"data_type: {type}")
        docs = self.getPath(type)
        t_lst = []
        t_c_lst = []
        for filepath, category_name in docs:
            lst_data = self.getDataForClass(filepath=filepath, class_name=category_name)
            total_class_freq = nltk.FreqDist({})
            # freqs on data-dim
            for freq, class_name in lst_data:
                total_class_freq.update(freq)

            n_lst_data = []
            # convert freqs to class-dim
            for freq, class_name in lst_data:
                new_freq = nltk.FreqDist({})
                for i, key in enumerate(freq.keys()):
                    new_freq.update({key: total_class_freq[key]})
                n_lst_data.append((new_freq, class_name))


            t_lst += n_lst_data
            t_c_lst += [(total_class_freq, category_name)]

        t_X = [freq for freq, class_name in t_lst]
        t_y = [class_name for freq, class_name in t_lst]
        print("\n\n")
        return t_X, t_y, list(zip(t_X, t_y)), t_c_lst


    def getX(self, type):
        if type == TRAIN:
            return self.X_train
        elif type == TEST:
            return self.X_test
        elif type == DEV:
            return self.X_dev

    def getY(self, type):
        if type == TRAIN:
            return self.y_train
        elif type == TEST:
            return self.y_test
        elif type == DEV:
            return self.y_dev

    def getSamples(self, type):
        if type == TRAIN:
            return self.train_samples
        elif type == TEST:
            return self.test_samples
        elif type == DEV:
            return self.dev_samples

    def getClSamples(self, type):
        if type == TRAIN:
            return self.train_cl_samples
        elif type == TEST:
            return self.test_cl_samples
        elif type == DEV:
            return self.dev_cl_samples

    def getPath(self, type):
        if type == TRAIN:
            return [(TRAIN_PATH + category + TRAIN_EXT, category) for category in CATEGORY_TYPES]
        elif type == TEST:
            return [(TEST_PATH + category + TEST_EXT, category) for category in CATEGORY_TYPES]
        elif type == DEV:
            return [(DEV_PATH + category + DEV_EXT, category) for category in CATEGORY_TYPES]

    def runNaiveBayes(self, type):
        print("NAIVE BAYES CLASSIFIER:\n")
        classifier = SklearnClassifier(MultinomialNB())
        classifier.train(self.train_samples)
        y_res, y_true = self.classify_all(type, classifier)
        confusion = confusion_matrix(y_true, y_res)
        print(confusion)
        print("\n\n")
        self.printResults(y_res, y_true)
        for i, genre in enumerate(CATEGORY_TYPES):
            print(f"Classify {genre}")
            a = [freq.most_common(10) for freq, name in self.test_cl_samples if name == genre][0]
            print(a)
            y_res, y_true = self.classify_genre(type, classifier, genre)
            a = [freq.N() for freq, name in self.train_cl_samples if name == genre]
            print(f"number of data: {a[0]}")
            self.printResults(y_res, y_true)
            self.printConfusionMatrixByRow(confusion, i, fp=True, tp=True)

    def runSVC(self, type):
        print("SUPPORT VECTOR CLASSIFIER:\n")
        classifier = SklearnClassifier(LinearSVC(dual="auto"))
        self.normalize(TRAIN)
        self.normalize(type)
        classifier.train(self.train_samples)
        y_res, y_true = self.classify_all(type, classifier)
        confusion = confusion_matrix(y_true, y_res)
        print(confusion)
        print("\n\n")
        self.printResults(y_res, y_true)


        for i, genre in enumerate(CATEGORY_TYPES):
            print(f"Classify {genre}\n")
            y_res, y_true = self.classify_genre(type, classifier, genre)

            self.printResults(y_res, y_true)
            # self.printConfusionMatrixByRow(confusion, i, fp=True, tp=True)


    def normalize(self, type):
        samples = self.getSamples(type)
        for freq, name in samples:
            for key, val in freq.items():
                freq[key] = (val + 1) / (freq.N() + len(freq.keys()))
    def classify_genre(self, type, classifier, genre):
        samples = self.getSamples(type)
        x_true = [freq for freq, name in samples if name == genre]
        y_true = [genre] * len(x_true)
        y_res = classifier.classify_many(x_true)
        return y_res, y_true

    def classify_all(self, type, classifier):
        samples = self.getSamples(type)
        x_true = [freq for freq, class_name in samples]
        y_true = [class_name for freq, class_name in samples]
        y_res = classifier.classify_many(x_true)
        return y_res, y_true

    def printResults(self, y_res, y_):
        acc = accuracy_score(y_, y_res)
        print(f"acc: {acc}")
        config = ["weighted"]
        for c in config:
            print(f"- config: {c}")
            precision = precision_score(y_, y_res, average=c)
            f1 = f1_score(y_, y_res,average=c)
            recall = recall_score(y_, y_res,average=c, zero_division=False)
            print(f"-- recall: {round(recall, 2)}")
            print(f"-- precision: {round(precision, 2)}")
            # print(f"-- calc f1: {round(2*precision*recall/(precision + recall))}")
            print(f"-- f1: {round(f1, 2)}")


    def printConfusionMatrixByRow(self, confusion, row_index, fp=False, tp=False):
        print(f"CATEGORY: {CATEGORY_TYPES[row_index]} - index: {row_index}")
        print(self.train_cl_samples[row_index][0].most_common(10))
        # if fp:
        #     row = confusion[row_index]
        #     print("FalsePos:")
        #     for index, (category, value) in enumerate(zip(CATEGORY_TYPES, row)):
        #         if index == row_index:
        #             print(f"-- tp: {category}: {value}")
        #         else:
        #             print(f"-- fp: {category}: {value}")
        # if tp:
        #     column = [row[row_index] for row in confusion]
        #     print("FalseNeg:")
        #     for index, (category, value) in enumerate(zip(CATEGORY_TYPES, column)):
        #         if index == row_index:
        #             print(f"--  tp: {category}: {value}")
        #         else:
        #             print(f"--  fn: {category}: {value}")
        print("\n")








def save_classifier(classifier, filename):  # filename should end with .pickle and type(filename)=string
    with open(filename, "wb") as f:
        pickle.dump(classifier, f)
    return


def load_classifier(filename):  # filename should end with .pickle and type(filename)=string
    classifier_file = open(filename, "rb")
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier


if __name__ == "__main__":
    # You may add or delete global variables.
    module = Module()
    module.runNaiveBayes(TEST)
    module.runSVC(TEST)
