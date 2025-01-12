import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from transformers import pipeline
from transformers import LongformerTokenizer, LongformerForMultipleChoice
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
import torch
from tqdm import tqdm
from copy import deepcopy, copy
from spectral_explain.dataloader import get_dataset
from transformers import BitsAndBytesConfig
from nltk.tokenize import word_tokenize, sent_tokenize

class TextModel:
    """Class for any model."""

    def __init__(self):
        pass

    def inference(self, X):
        raise NotImplementedError()

class Drop:
    def __init__(self, task, num_explain, device, seed):
        super().__init__()
        self.explicands = get_dataset(task, num_explain, seed)
        self.device = device
        self.mask_token = '[UNK]'
        quantization_config = BitsAndBytesConfig(
        load_in_8bit=True)


        self.trained_model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b', 
                            device_map = self.device,  quantization_config=quantization_config)
        self.tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.trained_model.eval()
        self.batch_size = 512

    def set_explicand(self, explicand):
        self.explicand = explicand
        return len(explicand['input'])
    
   
    def inference(self, X):
        # X is the masking pattern
        input_strings = []
        outputs = []
       
        for index in X:
            input = word_tokenize(copy(self.explicand['original']))
            for i,word in enumerate(input):
                if index[i] == 0:
                    input[i] = self.mask_token
            input_strings.append(' '.join(input))
            #input_strings.append(input.join(' '))
            
            # for i, location in enumerate(self.explicand['locations']):
            #     if index[i] == 0:
            #         input = input[:location[0]] + self.mask_token + input[location[1]:]
            # input_strings.append(input)
        
        for i in tqdm(range(0, len(input_strings), self.batch_size)):
            batch = input_strings[i:i+self.batch_size]
            batch_prompt = []
            for input in batch:
                prompt = f' \nContext: {input}\nQuestion: {self.explicand['question']}'
                batch_prompt.append(prompt)
            inputs = self.tokenizer(batch_prompt, return_tensors='pt', padding=True, truncation=True).to(self.trained_model.device)
            with torch.no_grad():
                model_outputs = self.trained_model.generate(inputs["input_ids"],attention_mask=inputs["attention_mask"], max_new_tokens=6, output_scores=True, return_dict_in_generate=True)
            print(self.tokenizer.batch_decode(model_outputs['sequences'], skip_special_tokens=True))
            transition_scores = self.trained_model.compute_transition_scores(model_outputs['sequences'], model_outputs['scores'], normalize_logits=True) # shape: (batch_size, generated_tokens)
            input_length = inputs['input_ids'].shape[1] # shape: (batch_size)
            generated_tokens = model_outputs['sequences'][:, input_length:]
            transition_scores, generated_tokens = transition_scores.detach().cpu().numpy(), generated_tokens.detach().cpu().numpy()

            
            for j in range(len(generated_tokens)): # skip last token
                sequence_prob = 0.0
                generated_sequence = []
                for tok, score in zip(generated_tokens[j], transition_scores[j]):
                    if tok == self.tokenizer.convert_tokens_to_ids('<|eot_id|>'): 
                        break
                    if tok == self.tokenizer.eos_token_id: #ignore pad token (set to eos token_id)
                        break
                    if tok == self.tokenizer.convert_tokens_to_ids('\n'): #ignore new line token
                        continue
                    if tok == self.tokenizer.convert_tokens_to_ids('Answer'): #ignore answer token
                        continue
                    if tok == self.tokenizer.convert_tokens_to_ids(':'): #ignore answer token
                        continue
                    else:
                        generated_sequence.append(self.tokenizer.decode([tok]))
                        sequence_prob += score # add log probability of token
                sequence_prob = np.exp(sequence_prob)
                sequence_prob = np.log(sequence_prob/(1.0 - sequence_prob))
                if sequence_prob >= 1e9:
                    sequence_prob = 1e9
                if sequence_prob <= -1e9:
                    sequence_prob = -1e9
                #print(self.tokenizer.decode(generated_sequence))
                outputs.append(sequence_prob)
            del inputs, model_outputs, transition_scores, generated_tokens
            #torch.cuda.empty_cache()
        return outputs


            

        #     output_logits = model_outputs['scores'] # shape: (generated_tokens, batch_size, vocab_size)
        #     for j in range(len(model_outputs.shape[0])):
        #          outputs.append(1.0)
        #     #outputs[i:i+self.batch_size] = get_prob_of_answer(model_outputs)
        #     #outputs[i:i+self.batch_size] = self.trained_model.generate(**inputs, max_new_tokens=16)
        # return outputs

        
        

class Reviews:
    def __init__(self, task, num_explain, device, seed):
        super().__init__()
        self.explicands = get_dataset(task, num_explain, seed)
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


def get_model(task, num_explain=10, device=None, seed = 42):
    model = {
        "parkinsons": MLPClassification,
        "cancer": MLPClassification,
        "sentiment": Reviews,
        "sentiment_mini": Reviews,
        "drop": Drop,
    }.get(task, NotImplementedError())(task, num_explain, device, seed)

    return model.explicands, model

