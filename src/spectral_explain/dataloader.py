import shap
import pandas as pd
from sklearn.model_selection import train_test_split
import openml
import numpy as np
from sklearn.preprocessing import StandardScaler
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
import openml
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('stopwords')
nltk.download('punkt_tab')
import string
import json

def scaler_classification(X_train, X_test, y_train, y_test):
    s = StandardScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()


def scaler_regression(X_train, X_test, y_train, y_test):
    s = StandardScaler()
    X_train = s.fit_transform(X_train)
    X_test = s.transform(X_test)

    t = StandardScaler()
    y_train = t.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
    y_test = t.transform(y_test.to_numpy().reshape(-1, 1)).ravel()
    return X_train, X_test, y_train, y_test


class TabularDataset:
    """Class for any dataset."""

    def __init__(self):
        self.train_X = None

    def _get_masker(self, masker_type):
        return shap.maskers.Independent({
                                            "baseline_sample": self.train_X[0].reshape(
                                                (1, self.train_X.shape[1])),
                                            "baseline_median": np.median(self.train_X, axis=0).reshape(
                                                (1, self.train_X.shape[1])),
                                            "baseline_mean": np.mean(self.train_X, axis=0).reshape(
                                                (1, self.train_X.shape[1])),
                                            "baseline_zeros": np.zeros((1, self.train_X.shape[1])),
                                            "marginal": self.train_X,
                                        }.get(masker_type, NotImplementedError()))

    def retrieve(self, num_explain, mini):
        self.load()
        if num_explain > self.test_X.shape[0]:
            print(f'num_explain > test set size. Explaining all {self.test_X.shape[0]} test samples instead.')
            num_explain = self.test_X.shape[0]
        return self.train_X, self.train_y, self.test_X[:num_explain], self.masker


class Parkinsons(TabularDataset):
    """Large classification dataset (753 features) based on speech data.
    https://www.openml.org/search?type=data&id=42176&sort=runs&status=active
    """

    def __init__(self):
        super().__init__()
        self.name = 'parkinsons'
        self.task_type = 'classification'

    def load(self):
        masker_type = "baseline_mean"
        dataset = openml.datasets.get_dataset(42176)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        self.train_X, self.test_X, self.train_y, self.test_y = scaler_classification(
            *train_test_split(X, y, test_size=0.2, random_state=0))
        self.masker = self._get_masker(masker_type)


class Cancer(TabularDataset):
    """Medium classification dataset (30 features) to predict breast cancer prognosis.
    https://www.openml.org/search?type=data&sort=runs&id=1510&status=active
    """

    def __init__(self):
        super().__init__()
        self.name = 'cancer'
        self.task_type = 'classification'

    def load(self):
        masker_type = "baseline_mean"
        dataset = openml.datasets.get_dataset(1510)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        y = y.map(pd.Series({'1': 0, '2': 1}))
        self.train_X, self.test_X, self.train_y, self.test_y = scaler_classification(
            *train_test_split(X, y, test_size=0.2, random_state=0))
        self.masker = self._get_masker(masker_type)

class TextDataset:
    """Class for any dataset."""

    def __init__(self):
        self.documents = None

    def retrieve(self, num_explain, mini):
        self.load(mini)
        if num_explain > len(self.documents):
            print(f'num_explain > test set size. Explaining all {len(self.documents)} test samples instead.')
            num_explain = len(self.documents)
        return self.documents[:num_explain]


class Reviews(TextDataset):
    """First 100 rows of IMDB dataset
    https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download
    """

    def __init__(self):
        super().__init__()
        self.name = 'IMDBReviews'

    def load(self, mini):
        self.documents = []
        stop_words = set(stopwords.words('english'))
        stop_words.update(string.punctuation)
        filename = 'data/reviews-mini.txt' if mini else 'data/reviews.txt'
        with open(filename) as file:
            dataset = [line.rstrip() for line in file]

        for document in tqdm(dataset):
            # remove stopwords
            word_tokens = word_tokenize(document)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words][1:]
            locations = []
            substring = document
            cursor = 0
            for w in filtered_sentence:
                loc = substring[cursor:].find(w)
                locations.append((cursor + loc, cursor + loc + len(w)))
                cursor += loc + len(w)
            self.documents.append({'original': document, 'input': filtered_sentence, 'locations': locations})


class STS16(TextDataset):
    """First 100 test rows of STS16 sentence similarity dataset
    https://huggingface.co/datasets/mteb/sts16-sts
    """

    def __init__(self):
        super().__init__()
        self.name = 'STS16'

    def load(self, mini):
        self.documents = []
        filename = 'data/sts16-mini.txt' if mini else 'data/sts16.txt'
        with open(filename) as file:
            for line in file:
                sentence_dict = json.loads(line)
                original = sentence_dict['sentence1'] + ' ' + sentence_dict['sentence2']

                word_tokens = word_tokenize(original)

                # remove periods from word_tokens, used for splitting sentences
                word_tokens = [w for w in word_tokens if w != '.']

                locations = []
                substring = original
                cursor = 0
                for w in word_tokens:
                    loc = substring[cursor:].find(w)
                    locations.append((cursor + loc, cursor + loc + len(w)))
                    cursor += loc + len(w)

                self.documents.append({'original': original,
                                       'input': word_tokens,
                                       'locations': locations,
                                       'split_point': len(sentence_dict['sentence1'])})


class Race(TextDataset):
    """First 100 test rows of the RACE dataset from a high school level
    https://www.cs.cmu.edu/~glai1/data/race/
    """

    def __init__(self):
        super().__init__()
        self.name = 'RACE'

    def load(self, mini):
        self.documents = []
        stop_words = set(stopwords.words('english'))
        stop_words.update(string.punctuation)
        filename = 'data/race-mini.txt' if mini else 'data/race.txt'

        with open(filename) as file:
            for line in file:
                question_dict = json.loads(line)
                question = question_dict['questions'][0]
                options = question_dict['options'][0]
                context = question_dict['article']
                correct_answer = ord(question_dict['answers'][0]) - 65
                if mini:
                    filtered_context = sent_tokenize(context)
                else:
                    word_tokens = word_tokenize(context)
                    filtered_context = [w for w in word_tokens if not w.lower() in stop_words][1:]
                locations = []
                substring = context
                cursor = 0
                for w in filtered_context:
                    loc = substring[cursor:].find(w)
                    locations.append((cursor + loc, cursor + loc + len(w)))
                    cursor += loc + len(w)

                self.documents.append({'original': context,
                                       'input': filtered_context,
                                       'locations': locations,
                                       'question': question,
                                       'options': options,
                                       'correct_answer': correct_answer})


class MedQA(TextDataset):
    """First 100 rows of the training set of MedQA
    https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view
    """

    def __init__(self):
        super().__init__()
        self.name = 'MedQA'

    def load(self, mini):
        self.documents = []
        stop_words = set(stopwords.words('english'))
        stop_words.update(string.punctuation)

        with open('data/medqa.txt') as file:
            for line in file:
                question_dict = json.loads(line)
                # split question into context and question
                context_question_list = nltk.sent_tokenize(question_dict['question'])
                if '\n' in context_question_list[-1]:
                    context = ' '.join(context_question_list[:-1] + context_question_list[-1].split('\n')[:-1])
                    question = context_question_list[-1].split('\n')[-1]
                else:
                    context = ' '.join(context_question_list[:-1])
                    question = context_question_list[-1]

                options = [question_dict['options'][key] for key in ['A', 'B', 'C', 'D', 'E']]
                correct_answer = ord(question_dict['answer_idx']) - 65

                word_tokens = word_tokenize(context)
                words = [w for w in word_tokens if not w.lower() in stop_words][1:]
                locations = []
                substring = context
                cursor = 0
                for w in words:
                    loc = substring[cursor:].find(w)
                    locations.append((cursor + loc, cursor + loc + len(w)))
                    cursor += loc + len(w)

                self.documents.append({'original': context,
                                       'input': words,
                                       'locations': locations,
                                       'options': options,
                                       'correct_answer': correct_answer,
                                       'question': question})


def get_dataset(dataset, num_explain):
    mini = "mini" in dataset
    return {
        "parkinsons": Parkinsons,
        "cancer": Cancer,
        "sentiment": Reviews,
        "similarity": STS16,
        "comprehension": Race,
        "sentiment_mini": Reviews,
        "similarity_mini": STS16,
        "comprehension_mini": Race,
        "clinical": MedQA
    }.get(dataset, NotImplementedError())().retrieve(num_explain, mini)
