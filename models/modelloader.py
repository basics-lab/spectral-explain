import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from transformers import pipeline
from transformers import LongformerTokenizer, LongformerForMultipleChoice
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
import torch
from tqdm import tqdm
from data.dataloader import get_dataset


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
            if outputs[i][0]['label'] == 'positive':
                pos_logits[i] = outputs[i][0]['score']
            else:
                pos_logits[i] = outputs[i][1]['score']
        return pos_logits


class STS16:
    def __init__(self, task, num_explain, device):
        super().__init__()
        self.explicands = get_dataset(task, num_explain)
        self.device = device
        self.mask_token = '[UNK]'
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', torch_dtype=torch.float16).to(
            self.device)

    def set_explicand(self, explicand):
        self.explicand = explicand
        return len(explicand['input'])

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        unnormalized_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(unnormalized_embeddings, p=2, dim=1)

    def inference(self, X):
        input_strings = []
        for index in X:
            input = self.explicand['original']
            offset = 0
            for i, location in enumerate(self.explicand['locations']):
                if index[i] == 0:
                    input = input[:location[0] + offset] + self.mask_token + input[location[1] + offset:]
                    offset += len(self.mask_token) - (location[1] - location[0])
            input_strings.append(input[:self.explicand['split_point'] + offset])
            input_strings.append(input[self.explicand['split_point'] + offset:])

        batch_size = 50000
        similarities = np.zeros(X.shape[0])
        for i in range(0, 2 * X.shape[0], batch_size):
            batch = input_strings[i: i + batch_size]
            encoded_batch = self.tokenizer(batch, padding=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_batch)
            sentence_embeddings = self.mean_pooling(model_output,
                                                    encoded_batch['attention_mask']).detach().cpu().numpy()
            odd_rows = sentence_embeddings[1::2, :]
            even_rows = sentence_embeddings[::2, :]
            similarities[int(i / 2): int((i + batch_size) / 2)] = np.sum(odd_rows * even_rows, axis=1)
            del encoded_batch, model_output, sentence_embeddings
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return similarities


class Race:
    def __init__(self, task, num_explain, device):
        super().__init__()
        self.explicands = get_dataset(task, num_explain)
        model_path = "potsawee/longformer-large-4096-answering-race"
        self.tokenizer = LongformerTokenizer.from_pretrained(model_path)
        self.device = device
        self.mask_token = '<unk>'
        self.trained_model = LongformerForMultipleChoice.from_pretrained(model_path, torch_dtype=torch.float16).to(
            self.device)

    def set_explicand(self, explicand):
        self.explicand = explicand
        self.inputs = explicand['original']
        self.options = explicand['options']
        self.correct_answer = explicand['correct_answer']
        self.question = explicand['question']

        c_plus_q = explicand['original'] + ' ' + self.tokenizer.bos_token + ' ' + self.question
        self.max_length = self.tokenizer(
            c_plus_q, self.options[self.correct_answer],
            padding="longest",
            return_tensors='pt'
        )['input_ids'].shape[1] + len(explicand['locations'])
        return len(explicand['input'])

    def prepare_answering_input(self, context):
        c_plus_q = context + ' ' + self.tokenizer.bos_token + ' ' + self.question
        tokenized_examples = self.tokenizer(
            c_plus_q, self.options[self.correct_answer],
            padding="max_length",
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = tokenized_examples['input_ids'].unsqueeze(0)
        attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)
        example_encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return example_encoded

    def inference(self, X):
        batch_size = 12
        logits = np.zeros(X.shape[0])
        for i in tqdm(range(0, X.shape[0], batch_size)):
            inputs = []
            bs = len(X[i: i + batch_size, :])
            for j in range(bs):
                context = self.explicand['original']
                offset = 0
                for k, location in enumerate(self.explicand['locations']):
                    if X[i + j][k] == 0:
                        context = context[:location[0] + offset] + self.mask_token + context[location[1] + offset:]
                        offset += len(self.mask_token) - (location[1] - location[0])
                inputs.append(self.prepare_answering_input(context))

            input_ids_batch = torch.cat([inp['input_ids'] for inp in inputs], 0).to(self.device)
            attention_mask_batch = torch.cat([inp['attention_mask'] for inp in inputs], 0).to(self.device)
            outputs = self.trained_model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
            logits[i: i + bs] = outputs.logits[:, 0].detach().cpu().numpy()
            del input_ids_batch, attention_mask_batch, outputs
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        return logits


class MedQA:
    def __init__(self, task, num_explain, device):
        super().__init__()
        self.explicands = get_dataset(task, num_explain)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("m42-health/Llama3-Med42-8B")
        self.trained_model = AutoModelForCausalLM.from_pretrained("m42-health/Llama3-Med42-8B").to(self.device)
        self.trained_model.eval()

    def set_explicand(self, explicand):
        self.explicand = explicand
        self.inputs = explicand['original']
        self.options = explicand['options']
        self.correct_answer = explicand['correct_answer']
        self.question = explicand['question']
        self.mask_token = '<|eot_id|>'

        c_plus_q = explicand['original'] + ' ' + self.tokenizer.bos_token + ' ' + self.question
        self.max_length = self.tokenizer(
            c_plus_q,
            padding="longest",
            return_tensors='pt'
        )['input_ids'].shape[1] + len(explicand['locations'])

        self.correct_answer_tokens = self.tokenizer(
            self.options[self.correct_answer],
            padding="longest",
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids'].to(self.device)
        return len(explicand['input'])

    def inference(self, X):
        log_likelihoods = np.zeros(X.shape[0])
        for i, index in enumerate(X):
            context = self.explicand['original']
            offset = 0
            for i, location in enumerate(self.explicand['locations']):
                if index[i] == 0:
                    context = context[:location[0] + offset] + self.mask_token + context[location[1] + offset:]
                    offset += len(self.mask_token) - (location[1] - location[0])
            tokenized_input = self.tokenizer(context + ' ' + self.tokenizer.bos_token + ' ' + self.question).to(
                self.device)

            # compute log-likelihood of generating correct answer sequences
            log_sum = 0
            for i in range(self.correct_answer_tokens.shape[1]):
                # Predict with the given model
                with torch.no_grad():
                    outputs = self.trained_model(tokenized_input)
                    logit_predictions = outputs.logits

                # Extract the log probability of the most recently added token
                last_token_logit = logit_predictions[0, -1, :]
                last_token_log_probs = torch.nn.functional.log_softmax(last_token_logit, dim=-1)
                log_token_prob = last_token_log_probs[self.correct_answer_tokens[0, i]].item()
                log_sum += log_token_prob

                # Incrementally add an output token to the current sequence
                tokenized_input = torch.cat([tokenized_input, self.correct_answer_tokens[:, i:i + 1]], dim=1)

            log_likelihoods[i] = log_sum

            del tokenized_input
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        return log_likelihoods


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
        "similarity": STS16,
        "comprehension": Race,
        "sentiment_mini": Reviews,
        "similarity_mini": STS16,
        "comprehension_mini": Race,
        "clinical": MedQA
    }.get(task, NotImplementedError())(task, num_explain, device)

    return model.explicands, model
