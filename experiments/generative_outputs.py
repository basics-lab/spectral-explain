import json
import torch
import numpy as np
from spectral_explain.support_recovery import sampling_strategy
from transformers import AutoTokenizer, AutoModelForCausalLM
from experiment_utils import qsft_soft
from itertools import combinations
import os
import numba

numba.set_num_threads(8)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def reconstruct(recon_function, query_indices):
    beta_keys = list(recon_function.keys())
    beta_values = list(recon_function.values())
    freqs = np.array(query_indices) @ np.array(beta_keys).T
    H = np.exp(2j * np.pi * freqs / 2)
    y_hat = np.real(H @ np.array(beta_values))
    return y_hat


def remove_interactions(qsft_res, n, max=True):
    p = []

    for deg in range(0, 4):
        removals = list(combinations(range(n), deg))
        deg_rem_queries = np.ones((len(removals), n))
        for j, rem in enumerate(removals):
            for k in rem:
                deg_rem_queries[j][k] = 0
        rem_quant = reconstruct(qsft_res, deg_rem_queries)
        if max:
            loc = np.argmax(rem_quant)
        else:
            loc = np.argmin(rem_quant)
        p.append(deg_rem_queries[loc, :])
    return p

def measure_removal_changes(removal_locs, context_words, question):
    for removal_mask in removal_locs:
        masked_words = []
        for word_idx, orig_word in enumerate(context_words):
            if removal_mask[word_idx] > 0.5:
                masked_words.append(orig_word)
            else:
                masked_words.append('<unk>')
        masked_sentence = " ".join(masked_words)
        print(masked_sentence)
        model.set_explicand(masked_words, question)
        logit_diff = model.inference(np.ones((1000, n)), generative=True)
        print(f'True - False Logit: {logit_diff[0]}')
        print()


class Llama31:
    def __init__(self):
        # You must first request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
        # Then, log in to HuggingFace using 'huggingface-cli login' before usage
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        self.trained_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
                                                                  torch_dtype=torch.bfloat16,
                                                                  device_map="auto")
        self.trained_model.eval()

        self.mask_token = '<unk>'

    def set_explicand(self, context_words, question):
        self.context = context_words
        self.question = question

    def inference(self, X, generative=False):
        n = X.shape[1]
        outputs = np.zeros(len(X))

        if generative:
            count_true, count_false = 0, 0

        for j, index in enumerate(X):
            input = self.context.copy()
            for i in range(n):
                if index[i] == 0:
                    input[i] = self.mask_token

            system_message = "You are answering a true or false question. You will be given context and a question. Respond with only True or False, or else your question will be marked incorrect."
            user_message = f"Context: {' '.join(input)} \n Question: {self.question}"
            prompt = f"[System]: {system_message}\n[User]: {user_message}\n[Assistant]:"
            tokenized_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")

            output = self.trained_model(**tokenized_input)
            last_logits = output.logits[0, -1, :]
            true_logit = last_logits[2575].item()
            false_logit = last_logits[4139].item()
            outputs[j] = true_logit - false_logit

            if generative:
                output = self.trained_model.generate(**tokenized_input,
                                                     max_new_tokens=1,
                                                     pad_token_id=self.tokenizer.eos_token_id,
                                                     do_sample=True,
                                                     temperature=0.9,
                                                     top_k=2)
                generated_text_w_prompt = self.tokenizer.decode(output[0], skip_special_tokens=True)
                prompt_length = len(self.tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True))
                generated_text = generated_text_w_prompt[prompt_length:].strip()

                if generated_text == 'True':
                    count_true += 1
                elif generated_text == 'False':
                    count_false += 1

        if generative:
            print(f'Answer Counts --- True: {count_true}, False: {count_false}')


        return outputs


if __name__ == "__main__":
    MAX_B = 10
    MAX_ORDER = 6

    examples = json.load(open('data/fake_riddles.json'))
    model = Llama31()

    for example in examples:
        # Sample and reconstruct example using SpectralExplain
        context = example["context"]
        print(context)
        context_words = context.split(" ")
        question = example["question"]
        human_answer = example["human_answer"]
        choices = example["multiple_choice"]

        model.set_explicand(context_words, question)
        sampling_function = lambda X: model.inference(X)

        n = len(context_words)

        if not os.path.exists('samples/generative/'):
            os.makedirs('samples/generative/')
        save_dir = 'samples/generative/' + str(example["index"])

        signal, num_samples = sampling_strategy(sampling_function, MAX_B, n, save_dir, t=MAX_ORDER)
        qsft_result = qsft_soft(signal, MAX_B, MAX_ORDER)

        if example["correct_answer"] == "True":
            # Remove words to make answer more True
            print('Masking more True')
            measure_removal_changes(remove_interactions(qsft_result, n, max=True), context_words, question)
        else:
            # Remove words to make answer more False
            print('Masking more False')
            measure_removal_changes(remove_interactions(qsft_result, n, max=False), context_words, question)