import pandas as pd
from sklearn.model_selection import train_test_split
import openml
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from PIL import Image
import requests
import json
from itertools import islice

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
    """
    Class for handling tabular datasets.

    Methods:
    - _get_masker(masker_type): Get the masker for the dataset.
    - retrieve(num_explain): Retrieve the dataset and masker.
    """
    def __init__(self):
        self.train_X = None

    def retrieve(self, num_explain):
        self.load()
        if num_explain > self.test_X.shape[0]:
            print(f'num_explain > test set size. Explaining all {self.test_X.shape[0]} test samples instead.')
            num_explain = self.test_X.shape[0]
        return self.train_X, self.train_y, self.test_X[:num_explain], self.masker


class Parkinsons(TabularDataset):
    """
    Large classification dataset (753 features) based on speech data.
    https://www.openml.org/search?type=data&id=42176&sort=runs&status=active

    Methods:
    - load(): Load the dataset.
    """
    def __init__(self):
        super().__init__()
        self.name = 'parkinsons'
        self.task_type = 'classification'

    def load(self):
        dataset = openml.datasets.get_dataset(42176)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        self.train_X, self.test_X, self.train_y, self.test_y = scaler_classification(
            *train_test_split(X, y, test_size=0.2, random_state=0))
        self.masker = np.mean(self.train_X, axis=0)


class Cancer(TabularDataset):
    """
    Medium classification dataset (30 features) to predict breast cancer prognosis.
    https://www.openml.org/search?type=data&sort=runs&id=1510&status=active

    Methods:
    - load(): Load the dataset.
    """
    def __init__(self):
        super().__init__()
        self.name = 'cancer'
        self.task_type = 'classification'

    def load(self):
        dataset = openml.datasets.get_dataset(1510)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        y = y.map(pd.Series({'1': 0, '2': 1}))
        self.train_X, self.test_X, self.train_y, self.test_y = scaler_classification(
            *train_test_split(X, y, test_size=0.2, random_state=0))
        self.masker = np.mean(self.train_X, axis=0)

class TextDataset:
    """
    Class for handling text datasets.

    Methods:
    - retrieve(num_explain): Retrieve the dataset.
    """
    def __init__(self):
        self.documents = None

    def retrieve(self, num_explain):
        self.load()
        if num_explain > len(self.documents):
            print(f'num_explain > test set size. Explaining all {len(self.documents)} test samples instead.')
            num_explain = len(self.documents)
        return self.documents[:num_explain]


class Sentiment(TextDataset):
    """
    160 movie reviews from these two sources:
    https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download
    https://nlp.stanford.edu/sentiment/

    Methods:
    - load(): Load the dataset.
    """
    def __init__(self):
        super().__init__()
        self.name = 'IMDBReviews'

    def load(self):
        self.documents = []
        filename = 'data/sentiment.csv'

        reviews = pd.read_csv(filename)['review']
        dataset = []
        for r in reviews:
            dataset.append(r[1:-1])

        for document in tqdm(dataset):
            filtered_sentence = document.split()
            locations = []
            substring = document
            cursor = 0
            for w in filtered_sentence:
                loc = substring[cursor:].find(w)
                locations.append((cursor + loc, cursor + loc + len(w)))
                cursor += loc + len(w)
            self.documents.append({'original': document, 'input': filtered_sentence, 'locations': locations})

class Puzzles(TextDataset):
    """
    10 'fake' riddles from:
    https://github.com/autogenai/easy-problems-that-llms-get-wrong
    https://arxiv.org/abs/2405.19616

    Methods:
    - load(): Load the dataset.
    """
    def __init__(self):
        super().__init__()
        self.name = 'Riddles'

    def load(self):
        self.documents = []
        filename = 'data/puzzles.csv'

        riddles = pd.read_csv(filename)
        contexts = []
        for r in riddles['context']:
            contexts.append(r)


        for context, question in tqdm(zip(contexts, riddles['question'])):
            split_context = context.split()
            locations = []
            substring = context
            cursor = 0
            for w in split_context:
                loc = substring[cursor:].find(w)
                locations.append((cursor + loc, cursor + loc + len(w)))
                cursor += loc + len(w)
            self.documents.append({'original': context, 'input': split_context, 'locations': locations, 'question': question})

class VisualQA(TextDataset):
    """Visual question answering dataset
    """

    def __init__(self):
        super().__init__()
        self.name = 'VisualQA'

    def load(self):
        with open('data/visualqa.json', 'r') as file:
            self.documents = json.load(file)
            for doc in self.documents:
                doc["image"] = Image.open(requests.get(doc["image_url"], stream=True).raw).convert('RGB')     

def get_dataset(dataset, num_explain):
    """
    Get the dataset and masking pattern for the specified task.

    Parameters:
    - dataset: The name of the dataset.
    - num_explain: The number of examples to explain.

    Returns:
    - A tuple containing the training features, and masking pattern.
    """
    return {
        "parkinsons": Parkinsons,
        "cancer": Cancer,
        "sentiment": Sentiment,
        "puzzles": Puzzles,
        "visual-qa": VisualQA,
    }.get(dataset, NotImplementedError())().retrieve(num_explain)
