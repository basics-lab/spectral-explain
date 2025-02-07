import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from transformers import pipeline
import torch
from tqdm import tqdm
from spex.dataloader import get_dataset
from openai import OpenAI
import os

class MLPRegression:
    """
    A class to represent a Multi-Layer Perceptron (MLP) regression model.

    Attributes:
    - task: The task for which the model is trained.
    - num_explain: The number of examples to explain.
    - device: The device to run the model on (e.g., 'cpu', 'cuda').

    Methods:
    - set_explicand(explicand): Set the explicand for the model.
    - inference(X): Perform inference on the given input X.
    """
    def __init__(self, task, num_explain, device):
        super().__init__()
        self.train_X, self.train_y, self.explicands, self.masker = get_dataset(task, num_explain)
        n = self.train_X.shape[1]
        self.trained_model = MLPRegressor(hidden_layer_sizes=(n // 2, n // 4, n // 8),
                                          max_iter=500, random_state=0).fit(self.train_X, self.train_y.ravel())

    def set_explicand(self, explicand):
        self.explicand = explicand
        return self.train_X.shape[1]

    def inference(self, X):
        return self.trained_model.predict(np.where(X, self.explicand))


class MLPClassification:
    """
    A class to represent a Multi-Layer Perceptron (MLP) classification model.

    Attributes:
    - task: The task for which the model is trained.
    - num_explain: The number of examples to explain.
    - device: The device to run the model on (e.g., 'cpu', 'cuda').

    Methods:
    - set_explicand(explicand): Set the explicand for the model.
    - inference(X): Perform inference on the given input X.
    """
    def __init__(self, task, num_explain, device):
        super().__init__()
        self.train_X, self.train_y, self.explicands, self.masker = get_dataset(task, num_explain)
        n = self.train_X.shape[1]
        self.trained_model = MLPClassifier(hidden_layer_sizes=(n // 2, n // 4, n // 8),
                                           max_iter=500, random_state=0).fit(self.train_X, self.train_y.ravel())

    def set_explicand(self, explicand):
        self.explicand = explicand
        return self.train_X.shape[1]

    def inference(self, X):
        X = np.where(X, self.explicand, self.masker)
        if len(X.shape) == 1:
            X = X[None, :]
        predictions = self.trained_model.predict_proba(X)[:, 1]
        return np.log(predictions / (1 - predictions))

class Sentiment:
    """
    A class to represent a sentiment analysis model using a pre-trained transformer.

    Attributes:
    - task: The task for which the model is trained.
    - num_explain: The number of examples to explain.
    - device: The device to run the model on (e.g., 'cpu', 'cuda').

    Methods:
    - set_explicand(explicand): Set the explicand for the model.
    - inference(X): Perform inference on the given input X.
    """
    def __init__(self, task, num_explain, device):
        super().__init__()
        self.explicands = get_dataset(task, num_explain)
        self.device = device
        self.mask_token = '[UNK]'
        self.trained_model = pipeline("text-classification", model="lyrisha/distilbert-base-finetuned-sentiment",
                                      device=self.device, batch_size=2000, torch_dtype=torch.float16,
                                      top_k=None, function_to_apply='none', truncation=True)

    def set_explicand(self, explicand):
        self.explicand = explicand
        if len(explicand['input']) >= 256:
            batch_size = 500
        elif len(explicand['input']) >= 128:
            batch_size = 1000
        else:
            batch_size = 2000
        self.trained_model = pipeline("text-classification", model="lyrisha/distilbert-base-finetuned-sentiment",
                                      device=self.device, batch_size=batch_size, torch_dtype=torch.float16,
                                      top_k=None, function_to_apply='none', truncation=True)
        return len(explicand['input'])

    def inference(self, X):
        input_strings = []
        for index in X:
            input = self.explicand['original']
            offset = 0
            for i, location in enumerate(self.explicand['locations']):
                if index[i] == 0:
                    input = input[:location[0] + offset] + self.mask_token + input[location[1] + offset:]
                    offset += len(self.mask_token) - (location[1] - location[0])
            input_strings.append(input)
        pos_logits = np.zeros(len(X))
        outputs = self.trained_model.predict(input_strings)
        for i in range(len(X)):
            if outputs[i][0]['label'] == 'POSITIVE':
                pos_logits[i] = outputs[i][0]['score']
            else:
                pos_logits[i] = outputs[i][1]['score']
        return pos_logits

class Puzzles:
    """
    A class to use OpenAI's GPT4o-mini to solve variants of popular puzzles.

    Attributes:
    - task: The task for which the model is trained.
    - num_explain: The number of examples to explain.
    - device: The device to run the model on (e.g., 'cpu', 'cuda').

    Methods:
    - set_explicand(explicand): Set the explicand for the model.
    - inference(X): Perform inference on the given input X.
    """
    def __init__(self, task, num_explain, device):
        super().__init__()
        self.explicands = get_dataset(task, num_explain)
        self.device = device
        self.mask_token = '[UNK]'
        assert os.environ.get("OPENAI_API_KEY"), "set openai api key as environment variable under OPENAI_API_KEY"
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def set_explicand(self, explicand):
        self.explicand = explicand
        self.question = explicand['question']
        self.context = explicand['input']
        return len(explicand['input'])

    def inference(self, X):
        logprob_diffs = np.zeros(len(X))
        for i, sample in tqdm(enumerate(X)):
            masked_context = []
            for j, s in enumerate(sample):
                if s == 1:
                    masked_context.append(self.context[j])
                else:
                    masked_context.append(self.mask_token)
            output = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", "content":
                        "Answer with the one word True or False only. Any other answer will be marked incorrect."
                    },
                    {
                        "role": "user", "content": " ".join(masked_context) + "\n" + self.question,
                    },
                ],
                model="gpt-4o-mini-2024-07-18",
                logprobs=True,
                max_completion_tokens=1,
                top_logprobs=20
            )
            top_logprobs = output.choices[0].logprobs.content[0].top_logprobs
            true_logprob = None
            false_logprob = None
            for logprob in top_logprobs:
                if logprob.token == "True":
                    true_logprob = logprob.logprob
                elif logprob.token == "False":
                    false_logprob = logprob.logprob

            # using logprob =-100 if True/False not in the top 20
            if true_logprob is None and false_logprob is None:
                logprob_diffs[i] = 0
            elif true_logprob is None:
                logprob_diffs[i] = -100 - false_logprob
            elif false_logprob is None:
                logprob_diffs[i] = true_logprob - (-100)
            else:
                logprob_diffs[i] = true_logprob - false_logprob
        return logprob_diffs

def get_model(task, num_explain=10, device=None):
    """
    Get the model and explicands for the specified task.

    Parameters:
    - task: The task for which the model is trained.
    - num_explain: The number of examples to explain (default is 10).
    - device: The device to run the model on (e.g., 'cpu', 'cuda').

    Returns:
    - A tuple containing the explicands and the model.
    """
    model = {
        "parkinsons": MLPClassification,
        "cancer": MLPClassification,
        "sentiment": Sentiment,
        "puzzles": Puzzles
    }.get(task, NotImplementedError())(task, num_explain, device)

    return model.explicands, model
