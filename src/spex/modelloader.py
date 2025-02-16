import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from transformers import pipeline
import torch
from tqdm import tqdm
from openai import OpenAI
import os, copy
from PIL import Image, ImageDraw, ImageFilter
from itertools import islice
from transformers import AutoModelForCausalLM, AutoTokenizer
from spex.dataloader import get_dataset
import torch.nn.functional as F
def batched(iterable, batch_size):
    """
    Splits an iterable into batches
    """
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch
        
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

class VisualQA:
    """
    A class for performing Visual Question Answering (VQA) with a transformer-based model.
    
    Attributes:
        explicands (list): A list of explicand samples from the dataset.
        device (str): The device on which the model runs (e.g., 'cpu' or 'cuda').
        model (LlavaNextForConditionalGeneration): The transformer-based model for VQA.
        processor (AutoProcessor): The processor for handling input formatting.
    """
    
    def __init__(self, task, num_explain, device):
        super().__init__()
        self.explicands = get_dataset(task, num_explain)
        self.device = device
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
        ).to(device)
        torch.set_float32_matmul_precision('high')
        torch.set_num_threads(8)
        self.processor = AutoProcessor.from_pretrained(model_id)
        print("Model loaded.")
    
    def set_explicand(self, explicand, nx=4, ny=4):
        self.explicand = explicand
        self.image = explicand['image']
        self.question = explicand['question']
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.question},
                    {"type": "image"},
                ],
            },
        ]
        
        self.prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        self.target_output = explicand['target_output']
        self.nx = int(explicand['nx'])
        self.ny = int(explicand['ny'])
        self.n = self.nx * self.ny
        
        return self.n
    
    def mask_image(self, image: Image, mask_array):
        """
        Masks patches of an image by blurring the masked regions based on a grid mask array.
        
        Args:
            image (Image): The input image to mask.
            mask_array (list of lists): 2D list containing 0 or 1 to indicate masked (0) or unmasked (1) patches.
        
        Returns:
            Image: The masked image with blurred regions.
        """
        width, height = image.size
        cell_width = width // self.nx
        cell_height = height // self.ny

        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=100))
        mask = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(mask)

        for row in range(self.ny):
            for col in range(self.nx):
                if mask_array[row][col] == 0:
                    x0 = col * cell_width
                    y0 = row * cell_height
                    x1 = x0 + cell_width
                    y1 = y0 + cell_height
                    draw.rectangle([x0, y0, x1, y1], fill=0)
        
        return Image.composite(image, blurred_image, mask)
    
    def compute_log_prob(self, images, prompt_with_target, target_outputs):
        """
        Computes the log probabilities of the target outputs given the input images.
        
        Args:
            images (list of Image): List of masked images.
            prompt_with_target (list of str): List of input prompts concatenated with target output.
            target_outputs (list of str): List of expected target outputs.
        
        Returns:
            numpy.ndarray: Log probabilities of the target outputs.
        """
        with torch.inference_mode():
            inputs = self.processor(images=images, text=prompt_with_target, return_tensors='pt').to(self.device, torch.float16)
            outputs = self.model(**inputs)
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            del inputs, outputs

            target_inputs = self.processor.tokenizer(
                target_outputs, return_tensors='pt', add_special_tokens=False
            ).to(self.device)
            target_log_probs = log_probs[:, -target_inputs.input_ids.size(1) - 1 : -1, :]
            target_token_probs = target_log_probs.gather(2, target_inputs.input_ids.unsqueeze(-1)).squeeze(-1)
        
        return target_token_probs.sum(-1).detach().cpu().numpy()
    
    def inference(self, X):
        """
        Runs inference on a batch of input samples.
        
        Args:
            X (numpy.ndarray): Binary mask arrays representing different image masking configurations.
        
        Returns:
            numpy.ndarray: Log probabilities for each masked input sample.
        """
        batch_size = 16
        logits = np.zeros(X.shape[0])
        
        for batch_indices in batched(tqdm(range(0, X.shape[0])), batch_size):
            images = []
            prompt_with_target = []
            target_outputs = []
            
            for i in batch_indices:
                prompt_with_target.append(self.prompt + self.target_output)
                target_outputs.append(self.target_output)
                image_mask = X[i].reshape(self.ny, self.nx)
                masked_image = self.mask_image(self.image, image_mask)
                images.append(masked_image)
            
            logits[batch_indices] = self.compute_log_prob(images, prompt_with_target, target_outputs)
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return logits



class QAModel:
    """
    A class for performing question answering.
    """
    
    def __init__(self, task, num_explain, device, model_name = 'meta-llama/Llama-3.2-3B-Instruct', model_config = {'attn_implementation': 'flash_attention_2', 'torch_dtype': torch.float16}):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.mask_token = '<unk>'
        self.model.eval()
        self.device = device
        self.scalarizer = 'log_perplexity'
        self.explicands = get_dataset(task, num_explain)
        self.batch_size = 64
    
    def set_explicand(self, explicand):
        self.explicand = explicand
        self.context = explicand['input']
        self.question = explicand['question']
        self.answer = explicand['answer']
        self.n = len(self.context)
        self.get_original_output()
        return self.n
    
    def get_original_output(self):
        '''
        Computes the original output token ids'
        '''
        input_strings = [f"{self.explicand['question']}\nContext: {self.explicand['original']}\nAnswer:"]
        self.tokenizer.padding_side = 'left'
        inputs = self.tokenizer(input_strings, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True)
        original_output_token_ids = outputs['sequences'][:,inputs['input_ids'].shape[1]:][0].detach().cpu().numpy().tolist()
        original_output = self.tokenizer.decode(original_output_token_ids, skip_special_tokens=True)
        original_output_token_ids = [-1] + original_output_token_ids
        self.explicand['model_answer'] = original_output
        self.explicand['model_answer_token_ids'] = original_output_token_ids
        del outputs, inputs 

    def compute_log_prob(self, masked_inputs, original_output_token_ids):
        '''
        Computes the log probabilities of generating the original outputs given a masked context
        '''
        self.tokenizer.padding_side = 'left'
        batch_sequence_log_probs = [0.0] * len(masked_inputs)
        for j in range(len(original_output_token_ids) - 1):
            token_id = original_output_token_ids[j]
            prompts = []
            if token_id == -1:
                token_str = ''
            else:
                token_str = self.tokenizer.decode([token_id])
            for input in masked_inputs:
                prompts.append(f"{self.explicand['question']}\nContext: {input}\nAnswer: {token_str}")
            inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, output_scores=True, return_dict_in_generate=True, max_new_tokens=1)
            batch_token_probs = torch.stack(outputs['scores']).swapaxes(0,1)[:,0,:] # batch size * vocab size
            batch_token_probs = torch.clamp(batch_token_probs, min = 1e-6, max = 1.0 - 1e-6)
            batch_token_log_probs = F.log_softmax(batch_token_probs,dim = 1)[:,original_output_token_ids[j+1]].detach().cpu().numpy()
            batch_sequence_log_probs = [batch_sequence_log_probs[idx] + batch_token_log_probs[idx] for idx in range(len(masked_inputs))]
            del inputs, outputs, batch_token_probs, batch_token_log_probs
        
        return np.array(batch_sequence_log_probs)

    def create_prompt(self, input):
        prompt = ' '.join(input)
        print(prompt)
        return prompt
        
    def inference(self, X):
        input_strings = []
        outputs = [0.0] * len(X)

        for index in X:
            input = copy.copy(self.explicand['input'])
            for i, word in enumerate(self.explicand['input']):
                if index[i] == 0:
                    input[i] = self.mask_token
            input_strings.append(self.create_prompt(input))
        
        count = 0
        for i in tqdm(range(0, len(input_strings), self.batch_size)):
            batch = input_strings[i:i+self.batch_size]
            sequence_log_probs = self.compute_log_prob(batch, self.explicand['model_answer_token_ids'])
            outputs[count:count+len(sequence_log_probs)] = sequence_log_probs
            count += len(sequence_log_probs)
            del batch, sequence_log_probs
            torch.cuda.empty_cache()
        
        return np.array(outputs)


class HotPotQA(QAModel):
    """
    A class for performing HotpotQA.
    """
    def __init__(self, task, num_explain, device, model_name = 'meta-llama/Llama-3.2-3B-Instruct', model_config = {'attn_implementation': 'flash_attention_2', 'torch_dtype': torch.float16}):
        super().__init__(task, num_explain, device, model_name, model_config)

    def create_prompt(self, all_sentences):
        titles = self.explicand['titles']
        title_to_sent_id = self.explicand['title_to_sent_id']
        prompt = f"{self.explicand['question']}\n"
        for i in range(len(titles)):
            prompt += f"Title: {titles[i]} \n Context:"
            for sent_id in title_to_sent_id[titles[i]]:
                prompt += f"{all_sentences[sent_id]}"
            prompt += "\n"
        prompt += f"Answer: "
        print(prompt)
        return prompt 


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
        "puzzles": Puzzles,
        "visual-qa": VisualQA,
        "drop": QAModel,
        "hotpotqa": HotPotQA,
    }.get(task, NotImplementedError())(task, num_explain, device)

    return model.explicands, model
