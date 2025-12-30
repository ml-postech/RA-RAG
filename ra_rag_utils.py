import argparse
import os
import json
from tqdm import tqdm
import random
import numpy as np

from src.utils import inference
from src.utils import (
    em_score,
    eval_misalignment,
    extract_ctx_from_formatted_ctx,
)
from src.reliability_utils import (
    weighted_majority_voting,
    iterative_weighted_majority_voting,
)
from src.reliability_utils import (
    convert_to_valid_answer,
    EntailmentDeberta,
    get_semantic_ids,
    unify_answer,
)
from src.contriever_dataset import (
    dataloader,
)


def multi_source_weight_est(
    pipeline,
    model_config,
    multi_src_est_rag_dataset,
    batch_size,
    num_src,
    max_iter,
    tol,
    filtering_method,
    wandb=None,
    step=None,
    save_est_output=False,
    log_name=None,
    src_reliabilities=None,
    it=None,
    filtering_threshold=0.5,
):
    multi_src_data = []
    multi_src_outputs = []
    filtered_multi_src_outputs = []
    multi_src_target_gt = []
    multi_src_is_fact = []

    for src_idx in range(num_src):
        retrieval_info = multi_src_est_rag_dataset[f"s_{src_idx}"]
        retrieval_results = retrieval_info["total_ctxs"]
        total_gt = retrieval_info["total_gt"]
        target_gt = retrieval_info["total_target_answers"]
        is_fact = retrieval_info["total_is_fact"]
        hasanswer_top_k = retrieval_info["total_hasanswer_top_k"]
        total_questions = retrieval_info["total_questions"]

        multi_src_target_gt.append(target_gt)
        multi_src_is_fact.append(is_fact)

        d_loader = dataloader(retrieval_results, batch_size=batch_size)
        multi_src_data.append(d_loader)
        outputs = inference(
            pipeline,
            d_loader,
            temperature=model_config["temperature"],
            max_new_tokens=model_config["max_new_tokens"],
            do_sample=model_config["do_sample"],
            desc=f"src_{src_idx} inference",
        )
        multi_src_outputs.append(outputs)

        before_misalignment_matrix = eval_misalignment(
            outputs=outputs,
            gt=total_gt,
            target_answers=target_gt,
            hasanswer_topk=hasanswer_top_k,
        )
        print(
            f"Weight Estimation Phase / misalignment matrix of s_{src_idx} before\n {before_misalignment_matrix}"
        )

        ctx_lst = [
            extract_ctx_from_formatted_ctx(d[1]["content"]) for d in retrieval_results
        ]

        valid_outputs, score_result, filtered_idx = convert_to_valid_answer(
            outputs,
            ctx_lst,
            filtering_method=filtering_method,
            questions=total_questions,
            threshold=filtering_threshold,
        )
        filtered_multi_src_outputs.append(valid_outputs)

        after_misalignment_matrix = eval_misalignment(
            outputs=valid_outputs,
            gt=total_gt,
            target_answers=target_gt,
            hasanswer_topk=hasanswer_top_k,
        )
        print(
            f"Weight Estimation Phase / misalignment matrix of s_{src_idx} after\n {after_misalignment_matrix}"
        )
        gt_em_results = em_score(valid_outputs, total_gt) # EM results for golden answer regardless of the given retrieval results
        taregt_em_results = em_score(valid_outputs, target_gt) # EM results for target answer based on the given retrieval results, if there is no answer in the retrieved document, the em result is 0

        print(f"src_{src_idx} src reliability: {src_reliabilities[src_idx]}")
        print(f"src_{src_idx} gt_em_results: {gt_em_results}")
        print(f"src_{src_idx} target_em_results: {taregt_em_results}")
        print()

        if wandb:
            wandb.log(
                {f"estimation_phase/misalignment_matrix/src_{src_idx}": before_misalignment_matrix},
                step=step,
            )
            wandb.log(
                {f"estimation_phase/gt_em_results/src_{src_idx}": gt_em_results},
                step=step,
            )
            wandb.log(
                {f"estimation_phase/target_em_results/src_{src_idx}": taregt_em_results},
                step=step,
            )

    print()

    multi_src_outputs = list(zip(*multi_src_outputs))
    filtered_multi_src_outputs = list(zip(*filtered_multi_src_outputs))

    multi_src_target_gt = list(zip(*multi_src_target_gt))
    multi_src_is_fact = list(zip(*multi_src_is_fact))
    
    total_questions = multi_src_est_rag_dataset["s_0"]["total_questions"]
    entailment_model = EntailmentDeberta()

    print(f"="*60)
    print(f"Estimating source reliabilties")
    print(f"="*60)
    
    clustering_results = []

    for q_idx, outputs in enumerate(filtered_multi_src_outputs):
        results = get_semantic_ids(
            outputs, entailment_model, question=total_questions[q_idx]
        )
        clustering_results.append(results)

    unified_multi_src_outputs = unify_answer(
        filtered_multi_src_outputs, clustering_results
    )
    estimated_weight_lst, est_reliabilities = iterative_weighted_majority_voting(
        filtered_multi_src_outputs, num_src, max_iter, tol
    )

    delta = np.random.random(size=len(estimated_weight_lst)) * 1e-8
    estimated_weight_lst += delta  # prevent to have same value

    if save_est_output:
        est_info = {
            "multi_src_data": multi_src_data,
            "total_gt": total_gt,
            "multi_src_target_gt": multi_src_target_gt,
            "multi_src_outputs": multi_src_outputs,  # not filtered outputs
            "src_reliabilities": src_reliabilities,
            "est_reliabilities": est_reliabilities.tolist(),
            "estimated_weight_lst": estimated_weight_lst.tolist(),
        }

        file_path = f"output_logs/{log_name}/{it}/multi_src_est_outputs"
        os.makedirs(file_path, exist_ok=True)

        with open(f"{file_path}/results.json", "w") as f:
            json.dump(est_info, f, indent=4, ensure_ascii=False)

    return (
        estimated_weight_lst,
        est_reliabilities,
    )



# extract the questions idx that are not answered yet
def extract_data_idx(answer_info, top_k_src):
    results = []

    for idx, info in enumerate(answer_info.values()):
        if len(info["answers"]) < top_k_src:
            results.append(idx)

    return results


def store_answer(outputs, data_idx_lst, answer_info, w):
    idk = "i don't know"

    for i, data_idx in enumerate(data_idx_lst):
        ans = outputs[i]
        if ans != idk:
            answer_info[f"d_{data_idx}"]["answers"].append(ans)
            answer_info[f"d_{data_idx}"]["weights"].append(w)


def extract_multi_src_outputs(answer_info, top_k_src):
    multi_src_outputs = []
    weight_lst = []

    for idx, info in enumerate(answer_info.values()):
        cur_ans = info["answers"]
        cur_w = info["weights"]

        assert (
            len(cur_ans) <= top_k_src
        ), f"len(cur_ans): {cur_ans}, top_k: {top_k_src}, cur_ans: {cur_ans}"

        for _ in range(top_k_src - len(cur_ans)):
            cur_ans.append("N/A")
            cur_w.append(0)

        multi_src_outputs.append(cur_ans)
        weight_lst.append(cur_w)

    return multi_src_outputs, weight_lst


def extract_valid_output_idx(answer_info):
    results = []

    for idx, info in enumerate(answer_info.values()):
        if info["answers"]:
            results.append(idx)

    return results


def get_oracle_mv_wmv(multi_src_rag_dataset, num_src, oracle_weight):

    total_outputs = []

    for src_idx in range(num_src):
        retrieval_info = multi_src_rag_dataset[f"s_{src_idx}"]
        target_gt = retrieval_info["total_target_answers"]
        outputs = []

        for idx, ret in enumerate(retrieval_info["total_hasanswer_top_k"]):
            if ret == "irrelevant":
                outputs.append("i don't know")
            else:
                outputs.append(target_gt[idx][0])

        total_outputs.append(outputs)

    total_gt = multi_src_rag_dataset["s_0"]["total_gt"]

    multi_total_outputs = list(zip(*total_outputs))
    num_problem = len(multi_total_outputs)
    num_src = len(multi_src_rag_dataset)

    oracle_weight_lst = [oracle_weight for _ in range(num_problem)]

    mv_reliabilities = np.ones((num_problem, num_src))
    delta = np.random.random((num_problem, num_src)) * 1e-8
    mv_reliabilities += delta

    oracle_mv_estimated_answer = weighted_majority_voting(
        multi_total_outputs, reliabilities=mv_reliabilities
    )
    oracle_mv_em_results = em_score(oracle_mv_estimated_answer, total_gt)

    oralce_wmv_estimated_answer = weighted_majority_voting(
        multi_total_outputs, reliabilities=oracle_weight_lst
    )
    oralce_wmv_em_results = em_score(oralce_wmv_estimated_answer, total_gt)

    return oracle_mv_em_results, oralce_wmv_em_results


def multi_source_inference_mv(
    pipeline,
    model_config,
    multi_src_rag_dataset,
    num_src,
    batch_size,
    filtering_method,
    save_est_output=False,
    log_name=None,
    it=None,
    wandb=None,
    filtering_threshold=None,
):

    total_outputs = []

    for src_idx in range(num_src):
        retrieval_info = multi_src_rag_dataset[f"s_{src_idx}"]
        retrieval_results = retrieval_info["total_ctxs"]
        total_gt = retrieval_info["total_gt"]
        target_gt = retrieval_info["total_target_answers"]
        is_fact = retrieval_info["total_is_fact"]
        hasanswer_top_k = retrieval_info["total_hasanswer_top_k"]
        total_questions = retrieval_info["total_questions"]

        d_loader = dataloader(retrieval_results, batch_size=batch_size)
        outputs = inference(
            pipeline,
            d_loader,
            do_sample=model_config["do_sample"],
            temperature=model_config["temperature"],
            max_new_tokens=model_config["max_new_tokens"],
            desc=f"src_{src_idx} inference",
        )

        misalignment_matrix = eval_misalignment(
            outputs=outputs,
            gt=total_gt,
            target_answers=target_gt,
            hasanswer_topk=hasanswer_top_k,
        )
        print(
            f"Test Phase / misalignment matrix of s_{src_idx} before\n {misalignment_matrix}"
        )

        ctx_lst = [
            extract_ctx_from_formatted_ctx(d[1]["content"]) for d in retrieval_results
        ]
        valid_outputs, score_result, filtered_idx = convert_to_valid_answer(
            outputs, ctx_lst, filtering_method, filtering_threshold, total_questions
        )
        total_outputs.append(valid_outputs)

        misalignment_matrix = eval_misalignment(
            outputs=valid_outputs,
            gt=total_gt,
            target_answers=target_gt,
            hasanswer_topk=hasanswer_top_k,
        )
        print(
            f"Test Phase / misalignment matrix of s_{src_idx} after\n {misalignment_matrix}"
        )

        target_em_results = em_score(valid_outputs, target_gt)
        print(f"src_{src_idx} target_em_results: {target_em_results}")

        if wandb is not None:
            wandb.log(
                {f"Test_phase/src_{src_idx}/target_em_results": target_em_results},
                step=it,
            )

    total_gt = multi_src_rag_dataset["s_0"]["total_gt"]

    multi_total_outputs = list(zip(*total_outputs))
    num_problem = len(multi_total_outputs)
    num_src = len(multi_src_rag_dataset)

    mv_reliabilities = np.ones((num_problem, num_src))
    delta = np.random.random((num_problem, num_src)) * 1e-6
    mv_reliabilities += delta

    entailment_model = EntailmentDeberta()

    # Estimating reliabilties for unfiltered outputs
    unfiltered_clustering_results = []

    for q_idx, outputs in enumerate(multi_total_outputs):
        results = get_semantic_ids(
            outputs, entailment_model, question=total_questions[q_idx]
        )
        unfiltered_clustering_results.append(results)

    unified_multi_src_outputs = unify_answer(
        multi_total_outputs, unfiltered_clustering_results
    )
    total_outputs = list(zip(*unified_multi_src_outputs))

    mv_estimated_answer = weighted_majority_voting(
        unified_multi_src_outputs, reliabilities=mv_reliabilities
    )
    mv_em_results = em_score(mv_estimated_answer, total_gt)

    if save_est_output:
        est_info = {
            "total_gt": total_gt,
            "multi_src_outputs": multi_total_outputs,
            "total_estimated_answers": mv_estimated_answer,
        }
        est_info["unified_multi_src_outputs"] = unified_multi_src_outputs
        file_path = f"output_logs/{log_name}/{it}/multi_src_test_outputs"
        os.makedirs(file_path, exist_ok=True)

        with open(f"{file_path}/results.json", "w") as f:
            json.dump(est_info, f, indent=4, ensure_ascii=False)

    return total_outputs, total_gt, mv_em_results


def mv_to_wmv(total_outputs, total_gt, estimated_weight_lst, top_k_src):
    num_src = len(total_outputs)
    src_idx_lst = list(range(num_src))
    src_idx_lst = [
        i for _, i in sorted(zip(estimated_weight_lst, src_idx_lst), reverse=True)
    ]

    num_problem = len(total_outputs[0])

    wmv_outputs = []
    wmv_weight_lst = []

    for p_idx in range(num_problem):
        results = []
        weights = []
        for src_idx in src_idx_lst:
            if total_outputs[src_idx][p_idx] != "i don't know":
                results.append(total_outputs[src_idx][p_idx])
                weights.append(estimated_weight_lst[src_idx])

            if len(results) == top_k_src:
                break

        if len(results) != top_k_src:
            for _ in range(top_k_src - len(results)):
                results.append("i don't know")

        wmv_outputs.append(results)
        wmv_weight_lst.append(weights)

    estimated_answer = weighted_majority_voting(
        wmv_outputs, reliabilities=wmv_weight_lst
    )
    em_results = em_score(estimated_answer, total_gt)

    return em_results


def reliablity_aware_inference(
    pipeline,
    model_config,
    multi_src_retrieval_results,
    num_src,
    top_k_src,
    batch_size,
    estimated_weight_lst,
):
    src_idx_lst = list(range(num_src))
    top_k_src = min(num_src, top_k_src)

    src_idx_lst = [
        i for _, i in sorted(zip(estimated_weight_lst, src_idx_lst), reverse=True)
    ]

    num_data = len(multi_src_retrieval_results[f"s_0"]["total_ctxs"])
    answer_info = {f"d_{i}": {"answers": [], "weights": []} for i in range(num_data)}

    total_gt = multi_src_retrieval_results["s_0"]["total_gt"]
    total_questions = multi_src_retrieval_results["s_0"]["total_questions"]
    print(f"ordered src_idx: {src_idx_lst}\n")

    for ii, src_idx in enumerate(src_idx_lst):
        data_idx_lst = extract_data_idx(answer_info=answer_info, top_k_src=top_k_src)

        if not data_idx_lst:
            break

        retrieval_info = multi_src_retrieval_results[f"s_{src_idx}"]
        retrieval_results = retrieval_info["total_ctxs"]
        target_gt = retrieval_info["total_target_answers"]

        extracted_retrieval_results = [retrieval_results[i] for i in data_idx_lst]
        extracted_target_gt = [target_gt[i] for i in data_idx_lst]

        d_loader = dataloader(extracted_retrieval_results, batch_size=batch_size)
        outputs = inference(
            pipeline,
            d_loader,
            do_sample=model_config["do_sample"],
            temperature=model_config["temperature"],
            desc=f"src_{src_idx} inference",
        )

        cur_w = estimated_weight_lst[src_idx]
        store_answer(
            outputs=outputs, data_idx_lst=data_idx_lst, answer_info=answer_info, w=cur_w
        )

        em_results = em_score(outputs, extracted_target_gt)
        print(f"src_{src_idx} em_results: {em_results}")

    print()
    multi_src_outputs, weight_lst = extract_multi_src_outputs(answer_info, top_k_src)

    entailment_model = EntailmentDeberta()
    unfiltered_clustering_results = []

    for q_idx, outputs in enumerate(multi_src_outputs):
        results = get_semantic_ids(
            outputs, entailment_model, question=total_questions[q_idx]
        )
        unfiltered_clustering_results.append(results)

    unified_multi_src_outputs = unify_answer(
        multi_src_outputs, unfiltered_clustering_results
    )
    estimated_answer = weighted_majority_voting(
        unified_multi_src_outputs, weight_lst
    )
    em_results = em_score(estimated_answer, total_gt)

    return em_results


def baseline_inference(
    pipeline,
    model_config,
    multi_src_test_basline_retreival_results,
    batch_size,
    log_name=None,
    it=None,
):
    d_loader = dataloader(
        multi_src_test_basline_retreival_results["total_ctxs"], batch_size=batch_size
    )
    baseline_outputs = inference(
        pipeline,
        d_loader,
        do_sample=model_config["do_sample"],
        temperature=model_config["temperature"],
        max_new_tokens=model_config["max_new_tokens"],
        desc=f"baseline inference",
    )

    total_gt = multi_src_test_basline_retreival_results["total_gt"]
    total_target_answers = multi_src_test_basline_retreival_results[
        "total_target_answers"
    ]
    
    em_results = em_score(baseline_outputs, total_gt)
    target_em_results = em_score(baseline_outputs, total_target_answers)

    file_path = f"output_logs/{log_name}/{it}/baseline_outputs"
    os.makedirs(file_path, exist_ok=True)

    est_info = {
        "baseline_outputs": baseline_outputs,
        "total_target_answers": total_target_answers,
        "total_gt": total_gt,
    }

    with open(f"{file_path}/results.json", "w") as f:
        json.dump(est_info, f, indent=4, ensure_ascii=False)

    return em_results, target_em_results, baseline_outputs

