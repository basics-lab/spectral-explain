import shap
import pandas as pd
from sklearn.model_selection import train_test_split
import openml
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter


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

    def retrieve(self, num_explain, mini, seed, **kwargs):
        self.load(mini, seed, **kwargs)
        if num_explain > len(self.documents):
            print(f'num_explain > test set size. Explaining all {len(self.documents)} test samples instead.')
            num_explain = len(self.documents)
        return self.documents[:num_explain]
    

class HotpotQA(TextDataset):
    """HotpotQA dataset"""

    def __init__(self):
        super().__init__()
        self.name = 'hotpotqa'
        self.task = 'distractor'
        self.split = 'validation'

    def load(self, mini, seed):
        self.documents = None
        dataset = load_dataset('hotpot_qa', self.task, self.split)
        dataset = dataset.shuffle(seed = seed)
        documents = []
        for sample in dataset:
            if not (sample['answer'] == 'yes' or sample['answer'] == 'no'):
                continue
            all_sentences = []
            title_to_context = {sample['context']['title'][i]: sample['context']['sentences'][i] for i in range(len(sample['context']['title']))} # dictionary of title to context
            sent_to_loc = {}
            for par in sample['context']['sentences']:
                for sent in par:
                    all_sentences.append(sent)
            sent_to_loc = {sent: i for i, sent in enumerate(all_sentences)}
            question = f"Question: {sample['question']}. Only answer with yes or no."
            answer = sample['answer']
            documents.append({'original': original, 'input': context_words, 'locations': None, 'question': question, 'answer': answer, 'mask_level': 'sentence'
                              ,'max_new_tokens': 1,'model_name': 'meta-llama/Llama-3.2-1B-Instruct','model_batch_size': 128,'id': sample['query_id']})
        self.documents = documents

#limited max length and changed answer selection length
class Drop(TextDataset):
    """Drop dataset"""

    def __init__(self):
        super().__init__()
        self.name = 'DROP'
        self.task = None
        self.split = 'validation'
        self.mask_level = 'word'
        self.max_answer_length = 1
        self.model_batch_size = 128

    def load(self,mini, seed, max_length = 250):
        self.documents = None
        dataset = load_dataset('drop', name = self.task, split = self.split)
        dataset = dataset.shuffle(seed = seed)
        documents = []
        
        for sample in dataset:
            if self.mask_level == 'word':
                context_words = word_tokenize(sample['passage'])
            elif self.mask_level == 'sentence':
                context_words = sent_tokenize(sample['passage'])
            else:
                raise ValueError(f'Invalid mask level: {self.mask_level}')
            if len(context_words) > max_length:
                continue
            original = sample['passage']
            question = f"{sample['question']}. Provide shortest answer possible, long answers are penalized heavily."
            answer_list = sample['answers_spans']['spans']
            


            answer_counts = Counter(answer_list)
            max_count = max(answer_counts.values())
            most_frequent_answers = [ans for ans, count in answer_counts.items() if count == max_count]
            answer = most_frequent_answers[0]  # Choose the first one in case of a tie
            answer_length = len(answer.split(' '))
            if answer_length > self.max_answer_length:
                continue

            
            cursor = 0
            substring = sample['passage']
            locations = []
            for w in context_words:
                loc = substring[cursor:].find(w)
                locations.append((cursor + loc, cursor + loc + len(w)))
                cursor += loc + len(w)
            documents.append({'original': original, 'input': context_words, 'locations': locations, 
                              'question': question, 'answer': answer, 'n': len(context_words),
                              'mask_level': self.mask_level, 'id': sample['query_id']})
        self.documents = documents

class CNN(TextDataset):
    """CNN dataset"""

    def __init__(self):
        super().__init__()
        self.name = 'cnn'
        self.task = '3.0.0'
        self.split = 'validation'
        self.mask_level = 'sentence'
        self.max_new_tokens = 8
        #self.model_name = 'meta-llama/Llama-3.2-1B-Instruct'
        #self.model_name = 'HuggingFaceTB/SmolLM-135M'
        self.model_batch_size = 128
    def load(self, mini, seed):
        self.documents = None
        dataset = load_dataset('cnn_dailymail',name = self.task, split = self.split)
        dataset = dataset.shuffle(seed = seed)
        documents = []
        for sample in dataset:
            original = sample['article']
            if self.mask_level == 'word':
                context_words = word_tokenize(sample['article'])
            elif self.mask_level == 'sentence':
                context_words = sent_tokenize(sample['article'])
            else:
                raise ValueError(f'Invalid mask level: {self.mask_level}')
            question = f'Please summarize the article as succinctly.'
            locations = []
            cursor = 0
            for sent in context_words:
                loc = original[cursor:].find(sent)
                locations.append((cursor + loc, cursor + loc + len(sent)))
                cursor += loc + len(sent)
            documents.append({'original': original, 'input': context_words, 'locations': locations, 'question': question, 'mask_level': self.mask_level, 'max_new_tokens': self.max_new_tokens,'model_name': self.model_name,'model_batch_size': self.model_batch_size})
        self.documents = documents

class HotpotQA(TextDataset):
    """HotpotQA dataset"""

    def __init__(self):
        super().__init__()
        self.name = 'hotpotqa'
        self.task = 'distractor'
        self.split = 'validation'
        self.mask_level = 'sentence'
    
    def load(self, mini, seed):
        self.documents = None
        dataset = load_dataset('hotpot_qa', self.task, self.split)
        dataset = dataset.shuffle(seed = seed)
        documents = []
        for sample in dataset:
            pass

class Reviews(TextDataset):
    """120 movie reviews
    https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download
    """

    def __init__(self):
        super().__init__()
        self.name = 'IMDBReviews'

    def load(self, mini, seed):
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


def get_dataset(dataset, num_explain, seed = 42, **kwargs):
    mini = "mini" in dataset
    return {
        "parkinsons": Parkinsons,
        "cancer": Cancer,
        "sentiment": Reviews,
        "sentiment_mini": Reviews,
        "drop": Drop,
        "cnn": CNN,
    }.get(dataset, NotImplementedError())().retrieve(num_explain, mini, seed, **kwargs)



