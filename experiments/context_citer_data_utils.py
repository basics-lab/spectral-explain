import re
import pandas as pd
import numpy as np
import ast
from datasets import load_dataset, Dataset, load_from_disk

def check_yes_no(answer_list):
    if len(set(answer_list)) > 1:
        return False
    for answer in answer_list:
        if answer not in ['YES', 'NO']:
            return False
    return True

def convert_yes_no_to_int(answer):
    if answer == 'YES':
        return 1
    elif answer == 'NO':
        return 0
    else:
        return -1

def read_data(data_path = "data/", dataset_name = 'hotpot_qa', split = 'test',**kwargs):
    try:
        dataset = load_from_disk(f"{data_path}/{dataset_name}.hf")
        dataset = dataset[split]
        return dataset
    except:
        dataset = load_dataset(dataset_name, trust_remote_code=True,**kwargs)
        dataset.save_to_disk(f"{data_path}/{dataset_name}.hf")
        dataset = dataset[split]
    return dataset


def preprocess_hotpot(rows: Dataset, num_rows: int = 10, seed: int = 1, only_yes_no: bool = False):
    '''
    Preprocess the hotpot dataset for the context citation experiment.
    '''
    rows = rows.shuffle(seed = seed)
    prompts = []
    for row in rows:
        if len(prompts) >= num_rows:
            break
        if not (row['answer'] == 'yes' or row['answer'] == 'no'):
            continue
        prompt = {}
        prompt['id'] = row['id']
        title_to_context = {row['context']['title'][i]: row['context']['sentences'][i] for i in range(len(row['context']['title']))} # dictionary of title to context
        all_sentences = []
        for par in row['context']['sentences']:
            for sent in par:
                all_sentences.append(sent)
        supporting_titles = row['supporting_facts']['title'] # dictionary with keys title and sent_id
        supporting_sent_ids = row['supporting_facts']['sent_id']
        supporting_context = [(title,sent_id) for title,sent_id in zip(supporting_titles,supporting_sent_ids)]
        prompt['supporting_context'] = [] # list of tuples (title,sent_id,sentence)
        prompt['supporting_sent_ids'] = [] # list of absolute sentence ids
        for title,sent_id in supporting_context:
            try:
                prompt['supporting_context'].append((title,sent_id,title_to_context[title][sent_id]))
                prompt['supporting_sent_ids'].append(all_sentences.index(title_to_context[title][sent_id]))
            except:
                break 
        prompt['title'] = row['context']['title']
        prompt['context'] = row['context']['sentences']
        if only_yes_no:
            prompt['question'] = f"Question: {row['question']}. Only answer with yes or no."
        else:
            prompt['question'] = f"Question: {row['question']}. Provide a concise answer."
        prompt['answer'] = row['answer']
        prompts.append(prompt)
    return prompts
  

def preprocess_drop(rows: Dataset, num_rows: int = 10, seed: int = 42):
    rows = rows.shuffle(seed = seed)
    prompts = []
    for row in rows:
        if len(prompts) >= num_rows:
            break
        prompt = {}
        prompt['id'] = row['query_id']
        prompt['title'] = ''
        prompt['context'] = row['passage']
        prompt['question'] = f'Query: {row["question"]}. Provide a concise answer.'
        prompt['answer'] = row['answer']
        prompts.append(prompt)
    return prompts

def preprocess_tydiqa(rows: Dataset, num_rows: int = 10, seed: int = 42, only_yes_no: bool = False):
    '''
    Preprocess the tydiqa dataset for the context citation experiment.

    rows: Dataset, the tydiqa hugging face dataset
    num_rows: int, number of rows to keep
    seed: int, random seed

    Returns: 
    prompts: list, list of prompts
    '''
    rows =  rows.filter(lambda example: example["language"].startswith("eng"))
    rows = rows.add_column("id", list(range(len(rows))))
    rows = rows.shuffle(seed = seed)
    prompts = []
    for row_idx in range(len(rows)):
        if len(prompts) >= num_rows:
            break
        row = rows[row_idx]
        #only allow for yes/no questions
        if not check_yes_no(row['annotations']['yes_no_answer']):
            continue
        prompt = {}
        prompt['title'] = row['document_title']
        prompt['sent_id'] = None
        prompt['context'] = row['document_plaintext']
        if only_yes_no:
            prompt['question'] = f"Question: {row['question_text']}. Only say yes or no, nothing else."
        else:
            prompt['question'] = f"Question: {row['question_text']}. Provide a concise answer."
        prompt['answer'] = row['annotations']['yes_no_answer'][0].lower()
        prompts.append((prompt,row))
    return prompts



def preprocess_cnn(rows: Dataset, num_rows: int = 10, seed: int = 42):
    '''
    Preprocess the tydiqa dataset for the context citation experiment.

    rows: Dataset, the tydiqa hugging face dataset
    num_rows: int, number of rows to keep
    seed: int, random seed

    Returns: 
    prompts: list, list of prompts
    '''
    rows = rows.shuffle(seed = seed)
    prompts = []
    for row in rows:
        if len(prompts) >= num_rows:
            break
        context = row['article']
        question = f'Please summarize the article in up to three sentences'
        prompt = {}
        prompt['title'] = ''
        prompt['context'] = context
        prompt['question'] = question
        prompt_format =  f"Content: {context}\nQuery: {question}"
        prompt['prompt'] = prompt_format
        prompt['id'] = row['id']
        prompts.append(prompt)
    return prompts
    #prompt_format = f""





        # print(row['annotations']['yes_no_answer'])
        # if row['annotations']['yes_no_answer'] == 'YES' or row['annotations']['yes_no_answer'] == 'NO':
        #     pass
        # else:
        #     continue
        # question = row['question_text']
        # context = row['document_plaintext']
        # title = row['document_title']
        # question = f'Query: {question}. Please only provide one answer to the question.'
        # prompt = {}
        # prompt['title'] = title
        # prompt['context'] = context
        # prompt['question'] = question
        # prompt_format = f"Title: {title}\nContent: {context}\nQuery: {question}"
        # prompt['prompt'] = prompt_format
        # prompt['id'] = row['id']
        # prompts.append(prompt)
    #return prompts
    #prompt_format = f""


 #     if only_yes_no and row['answer'] != 'yes' and row['answer'] != 'no':
    #         continue
    #     prompt['id'] = row['id']
    #     context = row['context']
    #     try:
    #         processed_context = preprocess_hotpot_context(context)
    #         prompt['title'] = [processed_context[i]['title'] for i in range(len(processed_context))]
    #         prompt['context'] = [processed_context[i]['context'] for i in range(len(processed_context))]
    #         prompt['question'] = f'Question: {row['question']}. Please provide only one answer.'
    #         prompts.append(prompt)
    #     except:
    #         pass

    # return prompts



  #     for i,title in enumerate(supporting_titles):
    #         prompt['sent_id'] = supporting_sent_ids[i]
    #         prompt['context'] = title_to_context[title]
         
    #     #  for fact in supporting_facts:
    #     #      supporting_sentence = title_to_context[fact['title']][fact['sent_id']]


       
    #     #  for i,sentence_id in enumerate(answer_sentence_ids):
    #     #      answer_parag
             
             
         
    #     prompt['title'] = row['context']['title']
    #     prompt['context'] = row['context']['sentences']
    #     if only_yes_no:
    #         prompt['question'] = f"Question: {row['question']}. Only answer with yes or no."
    #     else:
    #         prompt['question'] = f"Question: {row['question']}. Provide a concise answer."
    #     prompt['answer'] = row['answer']
    #     prompts.append(prompt)
    # return prompts
# def create_hotpot_prompt(rows: Dataset, num_rows: int = 16, seed: int = 42):
#     rows = rows.shuffle(seed = seed).select(range(10*num_rows))
#     rows = rows.to_iterable_dataset()
#     prompts = []
#     for row in rows:
#         prompt_format = ""
#         question = row['question']
#         context = row['context']
#         if len(prompts) >= num_rows:
#             break
        
#         try: 
#             processed_context = preprocess_hotpot_context(context)
#             prompt_format += ""
#             for i in range(len(processed_context)):
#                 context_list = processed_context[i]["context"]
#                 context_string = ''.join(context_list)
#                 prompt_format += f'Title: {processed_context[i]["title"]} \nContent: {context_string} \n'
#                 prompt_format += f"\nQuery: {question}. Please provide only one answer."
#             prompts.append(prompt_format)
#         except:
#             pass

#     return prompts



        # prompt_format = ""
        # for i in range(len(processed_context)):
        #     context_list = processed_context[i]["context"]
        #     context_string = ''.join(context_list)
        #     prompt_format += f'Title: {processed_context[i]["title"]} \nContent: {context_string} \n'
        #     prompt_format += f"\nQuery: {question}. Please provide only one answer."
        # prompts.append(prompt_format)


# def preprocess_hotpot_context(raw_string):
#     title_pattern = r"'title': array\((\[.*?\]), dtype=object\)"
#     title_match = re.search(title_pattern, raw_string, re.DOTALL)
#     if title_match:
#         titles = eval(title_match.group(1))  # Convert string representation to list
#     else:
#         raise ValueError("Titles not found in the input string.")
    
#     # Extract the sentences
#     remove_sentences_regex = r"'sentences':.*"#: array\(\[.*\])\)"  #   , dtype=object\)"
#     sentences_string = re.search(remove_sentences_regex, raw_string, re.DOTALL)
#     sentences_string = sentences_string.group(0).replace("'sentences': ", "")
#     sentences_string = sentences_string.replace('array', '').replace('dtype=object', '').strip()[:-1]
#     sentences_string = ast.literal_eval(sentences_string)[0]
#     doc_list = [doc[0] for doc in sentences_string]
#     context_data = [{"title": title, "context": context} for title, context in zip(titles, doc_list)]
#     return context_data
