import shap
import pandas as pd
from sklearn.model_selection import train_test_split
import openml
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import copy
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import AutoTokenizer
from collections import Counter

# def bucket_lengths(documents, bins = [0, 64, 128, 256, 512, 1024, 2048]):
#     bucketed_documents = {i: [] for i in range(len(bins) - 1)}
#     for doc in documents:
#         for i in range(len(bins) - 1):
#             if bins[i] <= doc['n'] < bins[i + 1]:
#                 bucketed_documents[i].append(doc)
#                 if len(bucketed_documents[i]) < 5:
#                     bucketed_documents[i].append(doc)
#     bucketed_documents = [bucketed_documents[i][j] for i in range(len(bins) - 1) for j in range(len(bucketed_documents[i]))]
#     return bucketed_documents

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
        self.load(seed = seed)
        if num_explain > len(self.documents):
            print(f'num_explain > test set size. Explaining all {len(self.documents)} test samples instead.')
            num_explain = len(self.documents)
        return self.documents[:num_explain]
    


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
        #self.model_batch_size = 128
    
    def filter_by_length(self,documents,bins = [0, 64, 128, 256, 512, 1024, 2048], num_in_each_bin = 5):
        bucketed_documents = {i: [] for i in range(len(bins) - 1)}
        for doc in documents:
            for i in range(len(bins) - 1):
                if bins[i] <= doc['n'] < bins[i + 1]:
                    if len(bucketed_documents[i]) < num_in_each_bin:
                        bucketed_documents[i].append(doc)
        bucketed_documents = [bucketed_documents[i][j] for i in range(len(bins) - 1) for j in range(len(bucketed_documents[i]))]
        return bucketed_documents
        

    def load(self,seed = 42, min_length = 64, max_length = 128, mini = False, **kwargs):
        dataset = load_dataset('drop', name = self.task, split = self.split)
        dataset = dataset.shuffle(seed = seed)
        documents = []
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
        
        for sample in dataset:
            if self.mask_level == 'word':
                context_words = word_tokenize(sample['passage'])
            elif self.mask_level == 'sentence':
                context_words = sent_tokenize(sample['passage'])
            else:
                raise ValueError(f'Invalid mask level: {self.mask_level}')
            if (len(context_words) < min_length) or (len(context_words) > max_length):
                continue
            original = sample['passage']
            #question = f"{sample['question']}. Provide shortest answer possible, long answers are penalized heavily."
            question = f'Answer the following question: {sample["question"]} based on the context provided below. Provide the shortest answer possible, long answers are penalized heavily.'
            answer_list = sample['answers_spans']['spans']
            


            answer_counts = Counter(answer_list)
            max_count = max(answer_counts.values())
            most_frequent_answers = [ans for ans, count in answer_counts.items() if count == max_count]
            answer = most_frequent_answers[0]  # Choose the first one in case of a tie
            answer_token_ids = tokenizer(answer, return_tensors='pt').input_ids[0,1:].tolist()
            if len(answer_token_ids) > self.max_answer_length:
                continue
           
            #answer_length = len(answer.split(' '))
            #if answer_length > self.max_answer_length:
            #    continue

            
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



class HotpotQA(TextDataset):
    """HotpotQA dataset"""

    def __init__(self):
        super().__init__()
        self.name = 'hotpotqa'
        self.task = 'distractor'
        self.split = 'validation'
        self.mask_level = 'sentence'
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
        self.max_token_ids = 1

  

    def load(self,  seed = 42, min_length = 17, max_length = 32, max_token_length = 1000, mini = False,**kwargs):
        documents = []
        dataset = load_dataset('hotpot_qa', self.task, self.split, trust_remote_code = True)[self.split]
        dataset = dataset.shuffle(seed = seed)
        for sample in dataset:
            sample_id = sample['id']
            question = f'Answer the following question: {sample["question"]} based on the context provided below. Answer in a single word or phrase. Long answers are penalized heavily.'
            answer = sample['answer']
            answer_token_ids = self.tokenizer(answer, return_tensors='pt').input_ids[0,1:].tolist()
            if len(answer_token_ids) > self.max_token_ids:
                continue

            all_sentences = [sent for par in sample['context']['sentences'] for sent in par]
            sent_to_loc = {sent: i for i, sent in enumerate(all_sentences)}      
            if (len(all_sentences) < min_length) or (len(all_sentences) > max_length):
                continue

            # For each title, returns sentences associated with it.       
            title_to_sent_id = self._get_title_to_sent_id(sample,sent_to_loc)

            # Required sentences for the model to answer the question.     
            supporting_sentences = self._get_supporting_facts(sample,sent_to_loc)

            # Creates prompt for the model.
            original = self.create_prompt(sample['context']['title'], all_sentences, title_to_sent_id)
            if len(self.tokenizer(original, return_tensors='pt').input_ids[0,1:].tolist()) > max_token_length:
                continue
          
            documents.append({'answer': answer, 'original': original, 'input': all_sentences, 'titles': sample['context']['title'],
                              'question': question, 'n': len(all_sentences), 'title_to_sent_id': title_to_sent_id,
                              'supporting_facts': supporting_sentences,'mask_level': 'sentence', 'id': sample_id})
        
        self.documents = documents
            
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

    def _get_supporting_facts(self, sample,sent_to_loc):
    
        # Title and sentence id of the supporting facts with respect to that title
        supporting_facts = [(sample['supporting_facts']['title'][i], sample['supporting_facts']['sent_id'][i]) for i 
                            in range(len(sample['supporting_facts']['title']))]
        supporting_sentences = []
        for title, sent_idx in supporting_facts:
            title_idx = sample['context']['title'].index(title)
            supporting_sentence = sample['context']['sentences'][title_idx][sent_idx]
            supporting_sentence_loc = sent_to_loc[supporting_sentence]
            supporting_sentences.append((title, supporting_sentence, supporting_sentence_loc))
        return supporting_sentences
    



def get_dataset(dataset, num_explain, seed = 42, **kwargs):
    mini = "mini" in dataset
    return {
        "parkinsons": Parkinsons,
        "cancer": Cancer,
        "sentiment": Reviews,
        "sentiment_mini": Reviews,
        "drop": Drop,
        "hotpotqa": HotpotQA,
    }.get(dataset, NotImplementedError())().retrieve(num_explain = num_explain, mini = mini, seed = seed, **kwargs)




# class CNN(TextDataset):
#     """CNN dataset"""

#     def __init__(self):
#         super().__init__()
#         self.name = 'cnn'
#         self.task = '3.0.0'
#         self.split = 'validation'
#         self.mask_level = 'sentence'
#         self.max_new_tokens = 8
#         #self.model_name = 'meta-llama/Llama-3.2-1B-Instruct'
#         #self.model_name = 'HuggingFaceTB/SmolLM-135M'
#         self.model_batch_size = 128
#     def load(self, mini, seed):
#         self.documents = None
#         dataset = load_dataset('cnn_dailymail',name = self.task, split = self.split)
#         dataset = dataset.shuffle(seed = seed)
#         documents = []
#         for sample in dataset:
#             original = sample['article']
#             if self.mask_level == 'word':
#                 context_words = word_tokenize(sample['article'])
#             elif self.mask_level == 'sentence':
#                 context_words = sent_tokenize(sample['article'])
#             else:
#                 raise ValueError(f'Invalid mask level: {self.mask_level}')
#             question = f'Please summarize the article as succinctly.'
#             locations = []
#             cursor = 0
#             for sent in context_words:
#                 loc = original[cursor:].find(sent)
#                 locations.append((cursor + loc, cursor + loc + len(sent)))
#                 cursor += loc + len(sent)
#             documents.append({'original': original, 'input': context_words, 'locations': locations, 'question': question, 'mask_level': self.mask_level, 'max_new_tokens': self.max_new_tokens,'model_name': self.model_name,'model_batch_size': self.model_batch_size})
#         self.documents = documents
