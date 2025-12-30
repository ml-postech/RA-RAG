import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import clean_str
import numpy as np

import os
import pickle
import torch
import json

from src.contriever_src import contriever
from src.contriever_src import normalize_text as contriever_normalize_text
from src.contriever_src.passage_retrieval import generate_retrieval_results
from src.prompts import convert_text_to_formatted, convert_text_to_gpt_format


def dataloader(dataset, batch_size=1):
    results = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    
    return results


def random_shuffle(x: list):
    """Randomly shuffle a list using numpy permutation."""
    return np.random.permutation(x).tolist()


def corpus_to_q_a(dataset,  is_valid=False):
    results = []
    data_matrix = dataset['data_matrix']
    q_info = dataset['q_info']
    
    for idx, data in enumerate(q_info):
        if is_valid and not data_matrix[idx]:
            continue
        
        target_answer = data['answers']
        target_answer = [clean_str(i) for i in target_answer]
        gt = data['gt']
        is_fact = data['is_fact']
        
        info = {
            'question': data['question'],
            'answers': target_answer,
            'gt': gt,
            'is_fact': is_fact
        }
        
        results.append(info)
    
    return results



def get_rag_dataset(retrieval_path, top_k, prompt_type, start_idx, end_idx):
    total_questions = []
    total_gt = []
    total_target_answers = []
    total_is_fact = []
    total_ctxs = []
    total_hasanswer_top_k = []
    
    with open(retrieval_path, 'r', encoding='utf-8') as f:
        retrieval_info = json.load(f)
    
    retrieval_info = retrieval_info[start_idx: end_idx]

    for idx, info in enumerate(retrieval_info):
        question = info['question']
        gt = info['gt']
        target_answers = info['answers']
        is_fact = info['is_fact']
        has_answer_top_k = info[f'hasanswer_top_{top_k}']
        ctxs = info['ctxs']
        top_k_ctxs = []
        
        for j in range(top_k):
            top_k_ctxs.append(ctxs[j]['text'])
        
        new_ctxs = convert_text_to_formatted(question=question, top_k_ctxs=top_k_ctxs, prompt_type=prompt_type)
        
        total_questions.append(question)
        total_gt.append(gt)
        total_target_answers.append(target_answers)
        total_is_fact.append(is_fact)
        total_hasanswer_top_k.append(has_answer_top_k)
        total_ctxs.append(new_ctxs)
    
    retrieval_result = {
        "total_questions": total_questions,
        "total_gt": total_gt,
        "total_target_answers": total_target_answers,
        "total_is_fact": total_is_fact,
        "total_hasanswer_top_k": total_hasanswer_top_k,
        "total_ctxs": total_ctxs,
    }
    
    return retrieval_result

def generate_corpus(dataset, src_reliability, p_rel, top_k=3):
    # p_rel: the probabilty of source for containing the relevant context

    q_info = []
    total_corpus = []
    data_matrix = np.zeros(len(dataset))
    
    for idx, data in enumerate(dataset):
        info = {}
        if np.random.random() > p_rel:
            info['question'] = data['question']
            info['question_idx'] = idx
            info['answers'] = ["i don\'t know"]
            gt = data['answers']
            gt = [clean_str(i) for i in gt]
            info['gt'] = gt
            info['is_fact'] = 'irrelevant'
            info['ctxs'] = []
            q_info.append(info)
            
            continue
        
        data_matrix[idx] = 1
        info = {}
        
        is_fact = np.random.random() < src_reliability
        ctxs_lst = []
        gt = data['answers']
        gt = [clean_str(i) for i in gt]
        
        if is_fact:
            ctxs = data['paraphrase']
            ctxs = random_shuffle(ctxs) 
            for j in range(top_k):
                ctxs_lst.append(ctxs[j])
            
            target_answer = data['answers']
            target_answer = [clean_str(i) for i in target_answer]
        else:
            ctxs = data['counterfactual']
            ctxs = random_shuffle(ctxs) 
            for j in range(top_k):
                ctxs_lst.append(ctxs[0]['contexts'][j])
            target_answer = [clean_str(ctxs[0]['answers'][0])]
        
        total_corpus += ctxs_lst
            
        info['question'] = data['question']
        info['question_idx'] = idx
        info['ctxs'] = ctxs_lst
        info['gt'] = gt
        info['answers'] = target_answer
        info['is_fact'] = is_fact
                
        q_info.append(info)
    
    
    corpus = {
        "q_info": q_info,
        "total_corpus": total_corpus,
        "data_matrix": data_matrix.tolist()
    }

    return corpus
    

def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, text in enumerate(passages):
            batch_ids.append(k)
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = contriever_normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings



def generate_embedding_passages(args, model, tokenizer, corpus, output_dir):
    passages = corpus['total_corpus']

    start_idx = 0
    end_idx = len(passages)
    
    passages = passages[start_idx:end_idx]
    print(f"Embedding generation for {len(passages)} passages from idx {start_idx} to {end_idx}.")

    allids, allembeddings = embed_passages(args, passages, model, tokenizer)

    save_file = os.path.join(output_dir, args.prefix + f"_{args.shard_id:02d}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {len(allids)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.")
    
    return save_file


def generate_multi_src_embedding(args, model, tokenizer, dataset, num_src, src_reliabilities, 
                                 log_name=None, top_k=3, output_dir=None, it=None, p_rel_lst=None):
    model, tokenizer, _ = contriever.load_retriever(args.contriever)
    print(f"Model loaded from {args.contriever}.", flush=True)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()
    
    multi_src_corpus = []
    multi_src_embedding_files = []
    
    for src_idx in range(num_src):
        print(f"Generating Retrieval Results of src_{src_idx} with reliabiltiy {src_reliabilities[src_idx]}")
        print()
        if p_rel_lst is not None:
            corpus = generate_corpus(dataset, src_reliabilities[src_idx], p_rel_lst[src_idx], top_k=top_k)
        else:
            corpus = generate_corpus(dataset, src_reliabilities[src_idx], args.p_rel, top_k=top_k)
        multi_src_corpus.append(corpus)
        embedding_output = f"{output_dir}/{log_name}/{it}/embeddings/src_{src_idx}"
        embedding_save_files = generate_embedding_passages(args=args, model=model, tokenizer=tokenizer, corpus=corpus, output_dir=embedding_output)
        multi_src_embedding_files.append(embedding_save_files)
    
    return multi_src_corpus, multi_src_embedding_files


def generate_multi_src_retrieval_results(args, model, tokenizer, multi_src_corpus, multi_src_embedding_files, 
                                         log_name, output_dir, q_info_lst, src_reliabilities, is_valid=False, it=None, top_k=3):
    assert type(q_info_lst) == list
    
    num_src = len(multi_src_corpus)
    val_name = "valid" if is_valid else "test"
    dir_path = f"{output_dir}/{log_name}/{it}/{val_name}_retrieval_results"
    os.makedirs(dir_path, exist_ok=True)
    
    with open(f"{output_dir}/{log_name}/{it}/src_reliabilities.json", 'w', encoding='utf-8') as f:
        json.dump(src_reliabilities, f)
    
    total_retrieval_output_paths = []
    multi_src_top_k_hits = []
    
    for src_idx in range(num_src):
        corpus = multi_src_corpus[src_idx]
        embedding_save_files = multi_src_embedding_files[src_idx]
        retrieval_output = os.path.join(dir_path, f"src_{src_idx}.json")
        total_retrieval_output_paths.append(retrieval_output)
        top_k_hits = generate_retrieval_results(args=args, model=model, tokenizer=tokenizer, 
                                   passages_embeddings=embedding_save_files, corpus=corpus, 
                                   output_path=retrieval_output, q_info=q_info_lst[src_idx], top_k=top_k)
        multi_src_top_k_hits.append(top_k_hits)
    
    return multi_src_top_k_hits, total_retrieval_output_paths


def integrate_multi_src_to_single_src(multi_src_corpus):
    integrated_q_info = []
    integrated_total_corpus = []
    
    num_src = len(multi_src_corpus)
    
    data_matrix = multi_src_corpus[0]['data_matrix']
    num_data = len(data_matrix)
    integrated_data_matrix = np.zeros(num_data)
    
    integrated_total_corpus = []
    
    for src_idx in range(num_src):
        corpus = multi_src_corpus[src_idx]
        total_corpus = corpus['total_corpus']
        integrated_total_corpus += total_corpus
    
    for data_idx in range(num_data):
        new_q_info = {
            'question': None,
            'question_idx': None,
            'ctxs': [],
            'gt': [],
            'answers': [],
            'is_fact': []
            }
        
        for src_idx in range(num_src):
            corpus = multi_src_corpus[src_idx]
            q_info = corpus['q_info'][data_idx]
            
            data_matrix = corpus['data_matrix']
                
            new_q_info['question'] = q_info['question']
            new_q_info['question_idx'] = q_info['question_idx']
            new_q_info['ctxs'] += q_info['ctxs']
            new_q_info['gt'] = q_info['gt']
            new_q_info['is_fact'].append(q_info['is_fact'])
            
            if "i don't know" not in q_info['answers']:
                new_q_info['answers'] += q_info['answers']
            
            if data_matrix[data_idx]:
                integrated_data_matrix[data_idx] = 1
        
        if not new_q_info['answers']:
            new_q_info['answers'].append("i don't know")
        
        integrated_q_info.append(new_q_info)            
    
    integrated_corpus = {
        'q_info': integrated_q_info,
        'total_corpus': integrated_total_corpus,
        'data_matrix': integrated_data_matrix
    }
    
    return integrated_corpus
    
    
    

def generate_baseline_embedding(args, model, tokenizer, corpus, log_name=None, output_dir=None, it=None):
    model, tokenizer, _ = contriever.load_retriever(args.contriever)
    print(f"Model loaded from {args.contriever}.", flush=True)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()
    
    embedding_output = f"{output_dir}/{log_name}/{it}/embeddings/baseline"
    embedding_save_files = generate_embedding_passages(args=args, model=model, tokenizer=tokenizer, corpus=corpus, output_dir=embedding_output)
    
    return embedding_save_files



def generate_baseline_retrieval_results(args, model, tokenizer, corpus, baseline_embedding_files, 
                                        log_name, output_dir, q_info, is_valid=False, it=None, top_k=10):
    
    val_name = "valid" if is_valid else "test"
    dir_path = f"{output_dir}/{log_name}/{it}/{val_name}_retrieval_results"
    os.makedirs(dir_path, exist_ok=True)

    retrieval_output = os.path.join(dir_path, f"baseline.json")
    top_k_hits = generate_retrieval_results(args=args, model=model, tokenizer=tokenizer, 
                                passages_embeddings=baseline_embedding_files, corpus=corpus, 
                                output_path=retrieval_output, q_info=q_info, top_k=top_k)
    
    
    return top_k_hits, retrieval_output
    

