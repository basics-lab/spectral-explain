import pandas as pd
from sklearn.model_selection import train_test_split
import openml
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from PIL import Image
import json
from itertools import islice
from nltk.tokenize import word_tokenize, sent_tokenize
from datasets import load_dataset
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
                doc["image"] = Image.open(doc["image_url"]).convert('RGB')  


class Drop(TextDataset):
    """
    Drop dataset for question answering
    We use the validation split of the Drop dataset and mask at the word level.
    """
    def __init__(self):
        super().__init__()
        self.name = 'Drop'
        self._split = 'validation'
    
    def retrieve(self, num_explain, buckets = {range(16,32): 1, range(32,64): 1, range(64,128): 1, range(128,256): 1, range(256,512): 1, range(512,1024): 1}):
        self.load()
        buckets_satisfied = {k: False for k in buckets}
        docs_to_return = []
        for doc in self.documents:
            for bucket in buckets:
                if doc['n'] in bucket and not buckets_satisfied[bucket]:
                    buckets_satisfied[bucket] = True
                    docs_to_return.append(doc)
            
        return docs_to_return
        # if num_explain > len(self.documents):
        #     print(f'num_explain > test set size. Explaining all {len(self.documents)} test samples instead.')
        #     num_explain = len(self.documents)
        # return self.documents[:num_explain]

    def _get_answer_counts(self, doc):
        answer_counts = Counter(doc['answers_spans']['spans'])
        max_count = max(answer_counts.values())
        most_frequent_answers = [ans for ans, count in answer_counts.items() if count == max_count]
        answer = most_frequent_answers[0]  # Choose the first one in case of a tie
        return answer

    def _remove_duplicates(self):
        seen_inputs = set()
        unique_documents = []
        for doc in self.documents:
            input_tuple = tuple(doc['input'])
            if input_tuple not in seen_inputs:
                seen_inputs.add(input_tuple)
                unique_documents.append(doc)
        self.documents = unique_documents

    def load(self):
        dataset = load_dataset('drop', name = None, split = self._split)
        self.documents = []
        for doc in dataset:
            context_words = word_tokenize(doc['passage'])
            original_context = doc['passage']
            question = f"Answer the following question based on the context provided below: {doc['question']}. Provide the shortest answer possible, long answers are not allowed."
            answer = self._get_answer_counts(doc)
            self.documents.append({'original': original_context, 'input': context_words, 'locations': [], 'question': question, 'answer': answer, 'n': len(context_words), 'id': doc['query_id']})
        self._remove_duplicates()
       
    
class HotpotQA(TextDataset):
    """
    HotpotQA dataset for question answering
    We use the validation split of the HotpotQA dataset and mask at the word level.
    """
    def __init__(self):
        super().__init__()
        self.name = 'hotpotqa'
        self.task = 'distractor'
        self._split = 'validation'

    
    def _remove_duplicates(self):
        seen_inputs = set()
        unique_documents = []
        for doc in self.documents:
            input_tuple = tuple(doc['input'])
            if input_tuple not in seen_inputs:
                seen_inputs.add(input_tuple)
                unique_documents.append(doc)
        self.documents = unique_documents
    
    def load(self,  seed = 42, mini = False,**kwargs):
        documents = []
        dataset = load_dataset('hotpot_qa', self.task, self._split, trust_remote_code = True)[self._split]
        dataset = dataset.shuffle(seed = seed)
        for sample in dataset:
            sample_id = sample['id']
            question = f'Answer the following question based on the context provided below: {sample["question"]}. Provide the shortest answer possible, long answers are not allowed.'
            answer = sample['answer']
            all_sentences = [sent for par in sample['context']['sentences'] for sent in par]
            sent_to_loc = {sent: i for i, sent in enumerate(all_sentences)}      
           
            # For each title, returns sentences associated with it.       
            title_to_sent_id = self._get_title_to_sent_id(sample,sent_to_loc)

            # Required sentences for the model to answer the question.     
            #supporting_sentences = self._get_supporting_facts(sample,sent_to_loc)

            # Creates prompt for the model.
            original = self.create_prompt(sample['context']['title'], all_sentences, title_to_sent_id)
           
            documents.append({'answer': answer, 'original': original, 'input': all_sentences, 'titles': sample['context']['title'],
                                'question': question, 'n': len(all_sentences), 'title_to_sent_id': title_to_sent_id, 'id': sample_id})

        
        self.documents = documents
        self._remove_duplicates()
    
    def create_prompt(self, titles, all_sentences, title_to_sent_id):
        prompt = ""
        for i in range(len(titles)):
            prompt += f"Title: {titles[i]} \n Context:"
            for sent_id in title_to_sent_id[titles[i]]:
                prompt += f"{all_sentences[sent_id]}"
            prompt += "\n"
        return prompt
     
    
    def _get_title_to_sent_id(self,sample,sent_to_loc):
        title_to_sent_id = {}
        for i in range(len(sample['context']['title'])):
            title = sample['context']['title'][i]
            title_to_sent_id[title] = []
            for sent in sample['context']['sentences'][i]:
                title_to_sent_id[title].append(sent_to_loc[sent])
        return title_to_sent_id

    # def _get_supporting_facts(self, sample,sent_to_loc):
    
    #     # Title and sentence id of the supporting facts with respect to that title
    #     supporting_facts = [(sample['supporting_facts']['title'][i], sample['supporting_facts']['sent_id'][i]) for i 
    #                         in range(len(sample['supporting_facts']['title']))]
    #     supporting_sentences = []
    #     for title, sent_idx in supporting_facts:
    #         title_idx = sample['context']['title'].index(title)
    #         supporting_sentence = sample['context']['sentences'][title_idx][sent_idx]
    #         supporting_sentence_loc = sent_to_loc[supporting_sentence]
    #         supporting_sentences.append((title, supporting_sentence, supporting_sentence_loc))
    #     return supporting_sentences
    




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
        "drop": Drop,
        "hotpotqa": HotpotQA,
    }.get(dataset, NotImplementedError())().retrieve(num_explain)
