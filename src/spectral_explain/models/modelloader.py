import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch
from tqdm import tqdm
from copy import copy
from spectral_explain.dataloader import get_dataset
from transformers import BitsAndBytesConfig
from nltk.tokenize import word_tokenize, sent_tokenize
import gc


quantization_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_compute_dtype=torch.bfloat16,
bnb_4bit_quant_type="nf4",
bnb_4bit_use_double_quant=True)

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

class TextModel:
    """Class for any model."""

    def __init__(self):
        pass

    def inference(self, X):
        raise NotImplementedError()

# see bos token issue and set max new tokens length
class QAModel:
    def __init__(self, device = 'auto', use_flash_attn = True, model_name = 'meta-llama/Llama-3.2-1B-Instruct', quantization_config = quantization_config, batch_size = 128):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        self.trained_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16,
                            device_map = self.device, quantization_config = quantization_config, 
                            attn_implementation = 'flash_attention_2' if use_flash_attn else None)#, attn_implementation = "flash_attention_2")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.mask_token = '<unk>'
        self.trained_model.eval()
        

    def set_explicand(self, explicand):
        self.explicand = explicand
        self.mask_level = explicand['mask_level']
        model_answer_toks = self.tokenizer(explicand['answer'])['input_ids'][1:]
        self.max_new_tokens = len(model_answer_toks)
        self.get_original_output(max_new_tokens = self.max_new_tokens)
        return len(explicand['input'])
    

    def get_original_output(self, max_new_tokens):
        #input_strings = [self.explicand['original']]
        #input_strings = [f"Context: {self.explicand['original']}\nQuestion: {self.explicand['question']}\nAnswer: "]
        input_strings = [f'{self.explicand['question']}\nContext: {self.explicand['original']}\nAnswer: ']
        inputs = self.tokenizer(input_strings, return_tensors='pt', padding=True, truncation=True).to(self.trained_model.device)
        with torch.no_grad():
            model_outputs = self.trained_model.generate(inputs["input_ids"],attention_mask=inputs["attention_mask"], 
                                                        length_penalty=1.0, do_sample=False, max_new_tokens= self.max_new_tokens, output_scores=True, 
                                                        pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True)
        original_output_token_ids = model_outputs['sequences'][:,inputs['input_ids'].shape[1]:][0].detach().cpu().numpy().tolist()
        original_decoded_output = self.tokenizer.decode(original_output_token_ids, skip_special_tokens=False,clean_up_tokenization_spaces=True)
        print(f'Original output: {original_decoded_output}')
        original_output_token_ids = [-1] + original_output_token_ids
        del model_outputs, inputs
        self.original_output_token_ids = original_output_token_ids
        self.original_decoded_output = original_decoded_output
        self.explicand['model_answer'] = original_decoded_output
        self.explicand['model_answer_token_ids'] = original_output_token_ids[1:]
        #return token_ids # shape is original answer tokens length

   
    
    def get_sequence_log_probs(self, batch_strings, original_output_token_ids):
        batch_sequence_log_probs = [0.0] * len(batch_strings)
        for j in range(len(original_output_token_ids) - 1):
            batch_prompt = []
            for input in batch_strings:
                if original_output_token_ids[j] == -1:
                    prompt =  f'{self.explicand['question']}\nContext: {input}\nAnswer: '
                else:
                    cur_answer = self.tokenizer.decode(original_output_token_ids[1:j+1], skip_special_tokens=False,clean_up_tokenization_spaces=True)
                    prompt =  f'{self.explicand['question']}\nContext: {input}\nAnswer: + {cur_answer}'
              
                batch_prompt.append(prompt)
            #print(batch_prompt[0])
            inputs = self.tokenizer(batch_prompt, return_tensors='pt', padding=True, truncation=True).to(self.trained_model.device)
            with torch.no_grad():
                model_outputs = self.trained_model.generate(inputs["input_ids"],attention_mask=inputs["attention_mask"], do_sample= False, max_new_tokens=1, output_scores=True, pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True)
            batch_token_probs = torch.stack(model_outputs['scores']).swapaxes(0,1)[:,0,:] # batch size * vocab size
            batch_token_probs = torch.clamp(batch_token_probs, min = 1e-6, max = 1.0 - 1e-6)
            batch_token_log_probs = F.log_softmax(batch_token_probs,dim = 1)[:,original_output_token_ids[j+1]].detach().cpu().numpy()
            batch_sequence_log_probs = [batch_sequence_log_probs[idx] + batch_token_log_probs[idx] for idx in range(len(batch_strings))]
            # token_probs = torch.stack(model_outputs['scores']).swapaxes(0,1)[:,0,original_output_token_ids[j+1]].detach().cpu().numpy()
            # print(token_probs.shape) # batch size * vocab size
            # token_probs = F.log_softmax(torch.stack(model_outputs['scores']).swapaxes(0,1)[:,0,:],dim = 1)[:,original_output_token_ids[j+1]].detach().cpu().numpy() # log probs of the next token (batch size * vocab size)
            # batch_sequence_log_probs = [batch_sequence_log_probs[tok_pos] + token_probs[tok_pos] for tok_pos in range(len(batch_strings))]
           
            del model_outputs, inputs, batch_prompt, batch_token_probs, batch_token_log_probs
            flush()

        batch_sequence_log_probs = np.array(batch_sequence_log_probs)
        return batch_sequence_log_probs
  
    def inference(self, X):
        # X is the masking pattern
        input_strings = []
        outputs = [0.0] * len(X)
       
        for index in X:
            if self.mask_level == 'sentence':
                input = sent_tokenize(copy(self.explicand['original']))
            elif self.mask_level == 'word':
                input = word_tokenize(copy(self.explicand['original']))
            else:
                raise ValueError(f'Invalid mask level: {self.mask_level}')
            for i,word in enumerate(input):
                if index[i] == 0:
                    input[i] = self.mask_token
            input_strings.append(' '.join(input))
        
        
        count = 0
        for i in tqdm(range(0, len(input_strings), self.batch_size)):
            batch = input_strings[i:i+self.batch_size]
            sequence_log_probs = self.get_sequence_log_probs(batch, self.original_output_token_ids)
            outputs[count:count+len(sequence_log_probs)] = sequence_log_probs
            count += len(sequence_log_probs)
            flush()

        return np.array(outputs)


class HotPotQAModel(QAModel):
    def __init__(self, device = 'auto', use_flash_attn = True, model_name = 'meta-llama/Llama-3.2-1B-Instruct', 
                quantization_config = quantization_config, batch_size = 128):
        super().__init__(device, use_flash_attn, model_name, quantization_config, batch_size)
    
    def create_prompt(self, all_sentences):
        titles = self.explicand['titles']
        title_to_sent_id = self.explicand['title_to_sent_id']
        prompt = [f'{self.explicand['question']}']
        for i in range(len(titles)):
            prompt += f"Title: {titles[i]} \n Context:"
            for sent_id in title_to_sent_id[titles[i]]:
                prompt += f"{all_sentences[sent_id]}"
            prompt += "\n"
        prompt += f"Answer: "
        return prompt

    def inference(self, X):
        # X is the masking pattern
        input_strings = []
        outputs = [0.0] * len(X)
       
        for index in X:
            input = copy(self.explicand['input'])
            for i, sent in enumerate(input):
                if index[i] == 0:
                    input[i] = self.mask_token
            
            input_strings.append(self.create_prompt(input))
            #input_strings.append(' '.join(input))
        
        
        count = 0
        for i in tqdm(range(0, len(input_strings), self.batch_size)):
            batch = input_strings[i:i+self.batch_size]
            sequence_log_probs = self.get_sequence_log_probs(batch, self.original_output_token_ids)
            outputs[count:count+len(sequence_log_probs)] = sequence_log_probs
            count += len(sequence_log_probs)
            flush()

        return np.array(outputs)

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
        "drop": QAModel,
        "cnn": QAModel,
    }.get(task, NotImplementedError())(task, num_explain, device, seed)

    return model.explicands, model

