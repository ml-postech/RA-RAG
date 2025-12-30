import numpy as np
from scipy.stats import beta
from collections import defaultdict
from tqdm import tqdm

from transformers import BartTokenizer, BartForConditionalGeneration
from copy import deepcopy

from Alignscore.alignscore import AlignScore

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def source_reliabilty_sampling(num_src, q_prior):
    alpha = 2*q_prior/(1-q_prior)
    beta = 2
    src_accs = np.random.beta(alpha, beta, size=num_src)
    return src_accs.tolist()


def adversary_hammer_sampling(num_src, num_adv):
    src_accs = []
    num_spammer = num_src - num_adv

    for _ in range(num_adv):
        src_accs.append(0.1)
    
    for _ in range(num_spammer):
        src_accs.append(0.9)
    
    return src_accs



def exist_majority(answers):
    check = defaultdict(int)
    
    for ans in answers:
        if ans == "i don't know":
            continue
        
        check[ans] += 1
    
    val = list(check.values())
    val = sorted(val, reverse=True)
    
    if not len(val):
        return False
    
    # single answer
    if len(val) == 1:
        return True
    
    # existing majority
    elif val[0] > val[1]:
        return True
    
    return False


def get_majority_sets(candidates):
    results = []
    for idx, cand in enumerate(candidates):
        if exist_majority(cand):
            results.append(cand)
    
    return results


def weighted_majority_voting(candidates, reliabilities):
    results = []
    
    for idx, cand in enumerate(candidates):
        weighted_votes = defaultdict(float)
        for answer, weight in zip(cand, reliabilities[idx]):
            if "i don't know" in answer:
                continue
            weighted_votes[answer] += weight
        
        if not len(weighted_votes):
            results.append("i don't know")
        else:
            estimated_answer = max(weighted_votes, key=weighted_votes.get)
            results.append(estimated_answer)
    return results


def update_reliability(candidates, estimated_answers):
    num_src = len(candidates[0])
    updated_reliabilities = np.array([.0 for _ in range(num_src)])
    src_cnt = np.array([.0 for _ in range(num_src)])
    
    for idx, cand in enumerate(candidates):
        for src_id, ans in enumerate(cand):
            if ans == "i don't know":
                continue
            
            src_cnt[src_id] += 1
            if estimated_answers[idx] == ans:
                updated_reliabilities[src_id] += 1
        
    updated_reliabilities = updated_reliabilities / src_cnt
    
    return updated_reliabilities


def iterative_weighted_majority_voting(outputs, num_src, max_iter, tol, stable_initialization=True):
    v = np.ones(num_src)
    
    for it in tqdm(range(max_iter), desc='Iterative weighted majority voting'):
        v_old = v.copy()
        v_tiled = np.tile(v_old, (len(outputs), 1))

        # For stable initialization, this option uses only output sets with agreement
        # (i.e., a clear majority) when estimating initial reliabilities.
        # If no consensus exists, an answer is selected at random because the added
        # small random noise in the initial reliability values can otherwise lead to
        # unstable initialization.

        if it == 0 and stable_initialization:
            valid_sets = get_majority_sets(outputs)
            if len(valid_sets) > 0:
                estimated_answer = weighted_majority_voting(valid_sets, v_tiled)
                updated_reliabilities = update_reliability(valid_sets, estimated_answer)
            else:
                estimated_answer = weighted_majority_voting(outputs, v_tiled)
                updated_reliabilities = update_reliability(outputs, estimated_answer)
        else:
            estimated_answer = weighted_majority_voting(outputs, v_tiled)
            updated_reliabilities = update_reliability(outputs, estimated_answer)

        v = num_src * updated_reliabilities - 1
        
        if np.linalg.norm(v_old - v) < tol:
            print(f'converged at iteration: {it}')
            print()
            break
    
    return v, updated_reliabilities


class QAConvertor():
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained("MarkS/bart-base-qa2d")
        self.model = BartForConditionalGeneration.from_pretrained("MarkS/bart-base-qa2d").to('cuda')
        
    def no_idk_filtering(self, answers):
        no_idk_idxs = []
        
        for idx, ans in enumerate(answers):
            if "i don't know" in ans:
                continue
            no_idk_idxs.append(idx)
        
        return no_idk_idxs
        
    
    def apply(self, questions, answers, batch_size=10):
        no_idk_idxs = self.no_idk_filtering(answers)
        bach_prompt = []
        
        for idx in no_idk_idxs:
            prompt = self.convert_prompt(questions[idx], answers[idx])
            bach_prompt.append(prompt)
        
        total_result = []
        
        for start_idx in tqdm(range(0, len(bach_prompt), batch_size), desc='convert outputs into declartive outputs'):
            cur_batch_promt = bach_prompt[start_idx:start_idx + batch_size]
            cur_batch_input = self.tokenizer(cur_batch_promt, return_tensors='pt', padding=True, truncation=True).to('cuda')
            cur_batch_output = self.model.generate(cur_batch_input.input_ids, max_new_tokens=200)
            result = self.tokenizer.batch_decode(cur_batch_output, skip_special_tokens=True)
            total_result += result
    
        converted_answers = deepcopy(answers)
        
        assert len(total_result) == len(no_idk_idxs)
        
        for idx, converted_output in enumerate(total_result):
            converted_answers[no_idk_idxs[idx]] = converted_output.lower()
        
        return converted_answers
    
    def convert_prompt(self, question, answer):
        prompt = f"question: {question} answer: {answer}"
        
        return prompt



class Filtering:
    def apply(self, output, context, questions=None):
        pass

class AlignScoreFiltering(Filtering):
    def __init__(self, threshold=0.1, ckpt_path=None) -> None:
        """
        Initialize AlignScore filtering.
        
        Args:
            threshold: Score threshold for filtering
            ckpt_path: Path to AlignScore checkpoint. If None, uses default path.
        """
        if ckpt_path is None:
            # Default checkpoint path relative to project root
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ckpt_path = os.path.join(project_root, 'Alignscore', 'ckpt_dir', 'AlignScore-base.ckpt')
        
        self.scorer = AlignScore(
            model='roberta-base', 
            batch_size=10, 
            device='cuda:0', 
            ckpt_path=ckpt_path, 
            evaluation_mode='nli_sp'
        )
        self.threshold = threshold
    
    def no_idk_filtering(self, answers):
        no_idk_idxs = []
        
        for idx, ans in enumerate(answers):
            if "i don't know" in ans:
                continue
            no_idk_idxs.append(idx)
        
        return no_idk_idxs

    def convert_answer_to_complete_answer(self, questions, outputs):
        convertor = QAConvertor()
        total_result = convertor.apply(questions, outputs)
    
        return total_result

    
    def apply(self, outputs, contexts, questions):
        print("\nConvert outputs to declarative_outputs\n")
        declarative_outputs = self.convert_answer_to_complete_answer(questions, outputs)
        print("Done\n")
        scores = self.scorer.score(contexts=contexts, claims=declarative_outputs)
        
        result = []
        score_result = []
        filtered_idx = []

        for idx, score in enumerate(scores):
            cur_output = outputs[idx]
            score_result.append(score)
            if score > self.threshold:
                result.append(cur_output)
            else:
                filtered_idx.append(idx)
                result.append("i don't know")
        
        return result, score_result, filtered_idx
                

def convert_to_valid_answer(outputs, ctx_lst, filtering_method, threshold, questions=None):
    print(f"\nApply filtering method: {filtering_method}\n")
    if filtering_method == 'align_score':
        assert questions is not None
        f_align = AlignScoreFiltering(threshold=threshold, ckpt_path="./Alignscore/ckpt_dir/AlignScore-base.ckpt")
        result = f_align.apply(outputs, ctx_lst, questions)
    else:
        raise NotImplementedError(f"{filtering_method} is not implemented")
    
    return result


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.device = "cuda"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(self.device)

    def check_implication(self, text1, text2, question, *args, **kwargs):
        inputs = self.tokenizer(question + " " + text1, question + " " + text2, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()

        return prediction



# def get_semantic_ids(strings_list, model, question, strict_entailment=False):
#     """Group list of predictions into semantic meaning."""
    
#     def are_equivalent(text1, text2, question):
    
#         implication_1 = model.check_implication(text1, text2, question)
#         implication_2 = model.check_implication(text2, text1, question)  # pylint: disable=arguments-out-of-order
        
#         assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

#         if strict_entailment:
#             semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

#         else:
#             implications = [implication_1, implication_2]
#             # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
#             semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
        
#         return semantically_equivalent
    
#     # Initialise all ids with -1.
#     semantic_set_ids = [-1] * len(strings_list)
#     # Keep track of current id.
#     next_id = 0
    
#     for i, string1 in enumerate(strings_list):
        
#         if string1 == "i don't know":
#             continue
        
#         # Check if string1 already has an id assigned.
#         if semantic_set_ids[i] == -1:
#             # If string1 has not been assigned an id, assign it next_id.
#             semantic_set_ids[i] = next_id
            
#             for j in range(i+1, len(strings_list)):
#                 if strings_list[j] == "i don't know":
#                     continue
                
#                 # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
#                 if are_equivalent(string1, strings_list[j], question):
#                     semantic_set_ids[j] = next_id
            
#             next_id += 1

#     return semantic_set_ids



def get_semantic_ids(strings_list, model, question, strict_entailment=False):
    """Group list of predictions into semantic meaning."""
    
    def are_equivalent(text1, text2, question):
    
        implication_1 = model.check_implication(text1, text2, question)
        implication_2 = model.check_implication(text2, text1, question)  # pylint: disable=arguments-out-of-order
        
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
        
        return semantically_equivalent
    
    def get_unique_strings(strings_list):
        unique_strings = []
        for string in strings_list:
            if string not in unique_strings:
                unique_strings.append(string)
        return unique_strings
    
    original_strings_list = deepcopy(strings_list)
    strings_list = get_unique_strings(strings_list) # Remove duplicate strings. This helps to avoid redundant computation.
    
    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    
    for i, string1 in enumerate(strings_list):
        
        if string1 == "i don't know":
            continue
        
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            
            for j in range(i+1, len(strings_list)):
                if strings_list[j] == "i don't know":
                    continue
                
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j], question):
                    semantic_set_ids[j] = next_id
            
            next_id += 1
    
    string_to_id = {}
    for idx, string in enumerate(strings_list):
        string_to_id[string] = semantic_set_ids[idx]

    original_semantic_set_ids = []
    for string in original_strings_list:
        original_semantic_set_ids.append(string_to_id[string])
    
    return original_semantic_set_ids


def unify_answer(multi_src_outputs, cluster_results):
    unified_outputs = []

    for idx1, cur_output in enumerate(multi_src_outputs):
        single_respones = {}
        cur_results = []
        
        cur_clutser_lst = cluster_results[idx1]

        for idx2, output in enumerate(cur_output):
            cluster_id = cur_clutser_lst[idx2]
            if cluster_id not in single_respones:
                single_respones[cluster_id] = output
            
            cur_results.append(single_respones[cluster_id])

        unified_outputs.append(cur_results)
    
    return unified_outputs