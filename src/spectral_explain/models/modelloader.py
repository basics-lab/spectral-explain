import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from transformers import pipeline
from transformers import LongformerTokenizer, LongformerForMultipleChoice
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
import torch
from tqdm import tqdm
from spectral_explain.dataloader import get_dataset


class TextModel:
    """Class for any model."""

    def __init__(self):
        pass

    def inference(self, X):
        raise NotImplementedError()


class Reviews:
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
        if len(explicand['input']) > 300:
            self.trained_model = pipeline("text-classification", model="lyrisha/distilbert-base-finetuned-sentiment",
                                          device=self.device, batch_size=500, torch_dtype=torch.float16,
                                          top_k=None, function_to_apply='none', truncation=True)
        elif len(explicand['input']) > 150:
            self.trained_model = pipeline("text-classification", model="lyrisha/distilbert-base-finetuned-sentiment",
                                          device=self.device, batch_size=1000, torch_dtype=torch.float16,
                                          top_k=None, function_to_apply='none', truncation=True)
        else:
            self.trained_model = pipeline("text-classification", model="lyrisha/distilbert-base-finetuned-sentiment",
                                      device=self.device, batch_size=2000, torch_dtype=torch.float16,
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

class TabularModel:
    """Class for any model."""

    def __init__(self):
        self.is_tree = False
        pass

    def inference(self, X):
        raise NotImplementedError()


class MLPRegression(TabularModel):
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
        print(len(X))
        return self.trained_model.predict(np.where(X, self.explicand, self.masker.data.flatten()))


class MLPClassification(TabularModel):
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
        X = np.where(X, self.explicand, self.masker.data.flatten())

        if len(X.shape) == 1:
            X = X[None, :]
        predictions = self.trained_model.predict_proba(X)[:, 1]
        return np.log(predictions / (1 - predictions))


def get_model(task, num_explain=10, device=None):
    model = {
        "parkinsons": MLPClassification,
        "cancer": MLPClassification,
        "sentiment": Reviews,
        "sentiment_mini": Reviews,
    }.get(task, NotImplementedError())(task, num_explain, device)

    return model.explicands, model
