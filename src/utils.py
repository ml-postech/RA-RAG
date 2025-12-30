import sys, os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json
import numpy as np
import random
import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

import re

from tqdm import tqdm
import datetime

import string
import glob




model_code_to_qmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

model_code_to_cmodel_name = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

def contriever_get_emb(model, input):
    return model(**input)

def dpr_get_emb(model, input):
    return model(**input).pooler_output

def ance_get_emb(model, input):
    input.pop('token_type_ids', None)
    return model(input)["sentence_embedding"]

    

def load_models(args):
    args.model_config_path = f'model_configs/{args.model_name}_config.json'
    model_config = load_json(args.model_config_path)
    model_name = model_config['model_name']
    model = AutoModelForCausalLM.from_pretrained( 
        model_name,
        device_map="cuda",   
        torch_dtype="auto",  
        trust_remote_code=True,  
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False
        
    )
    
    return pipeline


def load_beir_datasets(dataset_name, split, out_dir=None):
    assert dataset_name in ['nq', 'msmarco', 'hotpotqa']
    if dataset_name == 'msmarco': split = 'train'
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset_name)
    if not os.path.exists(data_path):
        data_path = util.download_and_unzip(url, out_dir)
    print(data_path)

    data = GenericDataLoader(data_path)
    if '-train' in data_path:
        split = 'train'
    corpus, queries, qrels = data.load(split=split)    

    return corpus, queries, qrels

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def save_results(results, dir, file_name="debug"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    if not os.path.exists(f'results/query_results/{dir}'):
        os.makedirs(f'results/query_results/{dir}', exist_ok=True)
    with open(os.path.join(f'results/query_results/{dir}', f'{file_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_results(file_name):
    with open(os.path.join('results', file_name)) as file:
        results = json.load(file)
    return results

def save_json(results, file_path="debug.json"):
    json_dict = json.dumps(results, cls=NpEncoder)
    dict_from_str = json.loads(json_dict)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict_from_str, f)

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    if len(s)>1 and s[-1] == ".":
        s=s[:-1]
    return s.lower()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|answer:)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def strip(text):
        return text.strip()
    
    def lower(text):
        return text.lower()

    result = white_space_fix(remove_articles(remove_punc(lower(strip(s)))))
    result.strip()
    
    return result



def extract_final_answer(sample, prompt_id=1):
    if prompt_id == 4:
        answers = [extract_final_answer(ans) for ans in sample]
        return answers
    
    sample = sample[0]['generated_text']    
    
    clean_ans = sample.lower().replace('answer:', "")
    clean_ans = re.sub(r"\b(a|an|the)\b", "", clean_ans)
    clean_ans = clean_ans.strip().lower()
    
    if len(clean_ans)>1 and clean_ans[-1] == ".":
        clean_ans=clean_ans[:-1]
        
    if not clean_ans:
        return "I don't know"
    
    return clean_ans
    


def em_score(outputs, answers):
    cnt = 0
    assert len(outputs) == len(answers), f"outputs: {len(outputs)} answers: {len(answers)}"
    
    for idx, (output, answer_lst) in enumerate(zip(outputs, answers)):
        if isinstance(output, set):
            found = False
            for ans in answer_lst:
                for o_ in output:
                    if ans in o_:
                        found = True
                        cnt += 1
                        break
                if found:
                    break
        elif isinstance(output, str):
            for ans in answer_lst:
                if ans in output:
                    cnt += 1
                    break
        else:
            raise NotImplementedError(f"output: {output}")
    
    return cnt / len(outputs)


def eval_misalignment(outputs, gt, target_answers, hasanswer_topk):
    confusion_matrix = np.zeros((3, 4)) # 3*4 matrix, (fact, counterfactual, irrelevant, hallucination)
    cnt_matrix = np.zeros(3) # (fact, counterfactual, irrelevant)
    
    for idx in range(len(outputs)):
        cur_output = outputs[idx]
        cur_gt = gt[idx]
        cur_target = target_answers[idx]
        cur_is_fact = hasanswer_topk[idx]

        if cur_is_fact == 'fact':
            flag = False
            cnt_matrix[0] += 1
            for o_ in cur_gt:
                if o_ in cur_output:
                    confusion_matrix[0][0] += 1
                    flag = True
                    break
            
            if not flag:
                if cur_output == "i don't know":
                    confusion_matrix[0][2] += 1
                else:
                    confusion_matrix[0][3] += 1

        elif cur_is_fact == 'counterfactual':
            flag = False
            cnt_matrix[1] += 1
            for o_ in cur_target:
                if o_ in cur_output:
                    confusion_matrix[1][1] += 1
                    flag = True
                    break
            
            if not flag:
                if cur_output == "i don't know":
                    confusion_matrix[1][2] += 1
                else:
                    flag = False
                    for o_ in cur_gt:
                        if o_ in cur_output:
                            confusion_matrix[1][0] += 1
                            flag = True
                            break
                    
                    if not flag:
                        confusion_matrix[1][3] += 1
        
        elif cur_is_fact == 'irrelevant':
            cnt_matrix[2] += 1
            if cur_output == "i don't know":
                confusion_matrix[2][2] += 1
            else:
                flag = False
                for o_ in cur_gt:
                    if o_ in cur_output:
                        confusion_matrix[2][0] += 1
                        flag = True
                        break
                
                if not flag:
                    confusion_matrix[2][3] += 1
        else:
            raise NotImplementedError(f"is_fact, {cur_is_fact}, not implemented")

    return confusion_matrix


def extract_ctx_from_formatted_ctx(ctx):
    start_index = ctx.find("\nContext: ") + len("\nContext: ")
    end_index = ctx.find("\nQuestion:")
    extracted_text = ctx[start_index:end_index].strip()
    return extracted_text



def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores


def inference(pipeline, query_lst, max_new_tokens=30, temperature=0.0, do_sample=True, desc=""):
    total_results = []

    for idx, query in enumerate(tqdm(query_lst, desc=desc)):
        responses = pipeline(
            query,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=pipeline.tokenizer.eos_token_id
        )
        final_responses = extract_final_answer(responses, prompt_id=4)
        
        total_results += final_responses
    
    return total_results 


def group_multi_source(outputs, num_src):
    assert len(outputs) % num_src == 0, f"len(outputs): {len(outputs)} num_src: {num_src}"
    
    total_results = []
    for idx in range(0, len(outputs), num_src):
        result = outputs[idx:idx+num_src]
        total_results.append(result)
    
    return total_results


def get_log_name(args):
    # Extract date and dataset name
    date_str = datetime.datetime.today().strftime('%y%m%d')
    dataset = os.path.splitext(os.path.basename(args.eval_dataset))[0]
    
    # Determine experiment case and adversary suffix
    exp_case = 'adv_hammer_prior' if args.adv_hammer else 'beta_prior'
    num_adv_suffix = f"_{args.num_adv}adv" if args.adv_hammer else ""
    
    # Common components for both log names
    common_parts = [
        dataset,
        exp_case + num_adv_suffix,
        f"num_src_{args.num_src}",
        f"p_rel_{args.p_rel}",
        f"q_prior_{args.q_prior}",
    ]
    
    # Build dataset log (shorter version)
    dataset_log_parts = common_parts + [
        f"top_k_{args.top_k}",
        f"exp_iter_{args.exp_iter}",
    ]
    dataset_log = "_".join(dataset_log_parts)
    
    # Build full log (includes more details)
    full_log_parts = [
        date_str,
        *common_parts,
        args.model_name,
        args.filtering_method,
        str(args.filtering_threshold),
        f"est_data_{args.num_est_data}",
        f"test_data_{args.num_test_data}",
        f"top_k_{args.top_k}",
        f"top_k_src_{args.top_k_src}",
        f"exp_iter_{args.exp_iter}",
    ]
    full_log = "_".join(full_log_parts)
    
    return full_log, dataset_log



def get_json_files(directory_path, pattern):
    if not os.path.isdir(directory_path):
        return None

    json_files = glob.glob(os.path.join(directory_path, pattern))
    json_files = sorted(json_files)
    return json_files
