import argparse
import os
import json
import numpy as np

from src.utils import load_models
from src.utils import load_json, setup_seeds, get_log_name, get_json_files
from src.reliability_utils import (
    source_reliabilty_sampling,
    adversary_hammer_sampling,
)
from src.contriever_src.contriever import load_retriever

from src.contriever_dataset import (
    generate_multi_src_retrieval_results,
    generate_multi_src_embedding,
    corpus_to_q_a,
    get_rag_dataset,
    integrate_multi_src_to_single_src,
    generate_baseline_embedding,
    generate_baseline_retrieval_results,
)

from ra_rag_utils import (
    multi_source_weight_est,
    baseline_inference,
    multi_source_inference_mv,
    mv_to_wmv,
    reliablity_aware_inference,
)


import torch
import wandb


def parse_args():
    """Parse command line arguments for the Reliability-Aware RAG system."""
    parser = argparse.ArgumentParser(
        description="Reliability-Aware RAG: Multi-source retrieval and inference system"
    )

    # Dataset
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Path to the evaluation dataset JSON file",
    )

    parser.add_argument("--project_name", default="Reliability-Aware RAG", type=str)

    # LLM settings
    parser.add_argument("--model_config_path", default=None, type=str)
    parser.add_argument("--model_name", type=str, default="llama3_8b")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_src", type=int, default=3)
    parser.add_argument("--num_est_data", type=int, default=100)
    parser.add_argument("--num_test_data", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3, help="The number of top k documents to retrieve for each query")
    parser.add_argument("--top_k_src", type=int, default=4, help="The number of top k sources for kappa-RRSS")

    parser.add_argument("--p_rel", type=float, default=0.6, help="the probability of the source containng relevant documents for the query")
    parser.add_argument("--q_prior", type=float, default=0.6, help="The prior probability for source reliability sampling")
    parser.add_argument("--prompt_type", type=str, default="rag", choices=['rag', 'llm'], help="The type of prompt to use for the RAG system. LLM does not use retrieval results.")

    parser.add_argument("--adv_hammer", action="store_true", default=False)
    parser.add_argument("--num_adv", type=int, default=3)
    parser.add_argument("--exp_iter", type=int, default=10)

    # Estimating the reliability of source
    parser.add_argument("--max_iter", type=int, default=100, help="The maximum number of iterations for the weight estimation")
    parser.add_argument("--tol", type=float, default=1e-20, help="The tolerance for the weight estimation")
    parser.add_argument("--save_est_output", action="store_true", default=False, help="If enabled, save the experimental results")
    
    parser.add_argument("--mv_to_wmv", action="store_true", help="If enabled, use the results of MV to generate the results of WMV. This avoids redundancy of computation by reusing the results of MV which is already computed.")

    parser.add_argument("--filtering_method", type=str, default="align_score")
    parser.add_argument("--filtering_threshold", type=float, default="0.5")

    # Contriever
    parser.add_argument("--contriever", type=str, default="facebook/contriever")
    parser.add_argument(
        "--retrieval_output_dir",
        type=str,
        default="./contriever_results",
        help="Directory path to save retrieval results and embeddings",
    )
    parser.add_argument(
        "--prefix", type=str, default="passages", help="prefix path to save embeddings"
    )
    parser.add_argument(
        "--shard_id", type=int, default=0, help="Id of the current shard"
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=512,
        help="Batch size for the passage encoder forward pass",
    )
    parser.add_argument(
        "--passage_maxlength",
        type=int,
        default=512,
        help="Maximum number of tokens in a passage",
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument(
        "--lowercase", action="store_true", help="lowercase text before encoding"
    )
    parser.add_argument(
        "--normalize_text", action="store_true", help="lowercase text before encoding"
    )
    parser.add_argument("--embedding_path", type=str, default="facebook/contriever")
    parser.add_argument(
        "--n_docs",
        type=int,
        default=10,
        help="Number of documents to retrieve per questions",
    )
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=1,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--save_or_load_index",
        action="store_true",
        help="If enabled, save index and load index if it exists",
    )
    parser.add_argument(
        "--question_maxlength",
        type=int,
        default=512,
        help="Maximum number of tokens in a question",
    )
    parser.add_argument(
        "--indexing_batch_size",
        type=int,
        default=1000000,
        help="Batch size of the number of passages indexed",
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument(
        "--n_bits", type=int, default=8, help="Number of bits per subquantizer"
    )

    # Setup
    parser.add_argument(
        "--score_function", type=str, default="dot", choices=["dot", "cos_sim"]
    )
    parser.add_argument(
        "--repeat_times",
        type=int,
        default=10,
        help="repeat several times to compute average",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=10,
        help="one of our parameters, the number of target queries",
    )
    
    # Setup
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--name", type=str, default="experiment", help="Name of the experiment for logging"
    )

    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    return args


def main():
    """Main function to run the Reliability-Aware RAG pipeline."""
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = "cuda"
    setup_seeds(args.seed)
    log_name, dataset_log = get_log_name(args)

    if args.wandb:
        wandb.init(project=args.project_name)
        wandb.run.name = log_name
        wandb.config.update(args)

    if args.model_config_path is None:
        args.model_config_path = f"model_configs/{args.model_name}_config.json"
    model_config = load_json(args.model_config_path)

    pipeline = load_models(args)

    # load contriever
    contriever, contriever_tokenizer, _ = load_retriever(args.contriever)
    print(f"Model loaded from {args.contriever}.", flush=True)
    print()

    contriever.eval()
    contriever = contriever.cuda()
    if not args.no_fp16:
        contriever = contriever.half()

    with open(args.eval_dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_mv_em_results = []
    total_wmv_em_results = []
    total_oracle_wmv_em_results = []

    total_base_em_results = []

    for it in range(args.exp_iter):
        print(f"{'='*60}")
        print(f"Iteration: {it+1}/{args.exp_iter}")
        print(f"{'='*60}\n")

        p_rel_lst = None
        if args.adv_hammer:
            print(f"Using adversary hammer prior")
            src_reliabilities = adversary_hammer_sampling(
                num_src=args.num_src, num_adv=args.num_adv
            )
        else:
            print(f"Using Beta prior")
            print(f"q_prior: {args.q_prior}")
            src_reliabilities = source_reliabilty_sampling(
                num_src=args.num_src, q_prior=args.q_prior
            )
            
        if args.wandb:
            for src_idx, val in enumerate(src_reliabilities):
                wandb.log({f"sampled reliabilties/src_{src_idx}": val}, step=it)

        dir_path = f"{args.retrieval_output_dir}/{dataset_log}/{it}"
        total_retrieval_output_paths = get_json_files(
            f"{dir_path}/retrieval_results", "src_*.json"
        )
        baseline_retrieval_output_path = get_json_files(
            f"{dir_path}/retrieval_results", "baseline.json"
        )

        if total_retrieval_output_paths and baseline_retrieval_output_path:
            print(f"{dir_path} exists")
            src_reliabilities_file = get_json_files(f"{dir_path}", "src_reliabilities.json")[0]
            src_reliabilities = load_json(src_reliabilities_file)
            print(f"sampled src_reliabilities: {src_reliabilities}\n")

            baseline_retrieval_output_path = baseline_retrieval_output_path[0]
        else:
            print(f"sampled src_reliabilities: {src_reliabilities}\n")
            os.makedirs(dir_path)
            multi_src_corpus, multi_src_embedding_files = generate_multi_src_embedding(
                args=args,
                model=contriever,
                tokenizer=contriever_tokenizer,
                dataset=dataset,
                num_src=args.num_src,
                src_reliabilities=src_reliabilities,
                log_name=dataset_log,
                top_k=args.top_k,
                output_dir=args.retrieval_output_dir,
                it=it,
                p_rel_lst=p_rel_lst,
            )
            
            # Validate retrieval results per source using validation questions where the source contains relevant information.
            print("\n\Generating Validation Retrieval Results")
            valid_q_info_lst = [
                corpus_to_q_a(d, is_valid=True) for d in multi_src_corpus
            ]
            valid_multi_src_top_k_hits, _ = generate_multi_src_retrieval_results(
                args=args,
                model=contriever,
                tokenizer=contriever_tokenizer,
                multi_src_corpus=multi_src_corpus,
                multi_src_embedding_files=multi_src_embedding_files,
                log_name=dataset_log,
                output_dir=args.retrieval_output_dir,
                q_info_lst=valid_q_info_lst,
                src_reliabilities=src_reliabilities,
                is_valid=True,
                it=it,
                top_k=args.top_k,
            )

            with open(f"{dir_path}/valid_multi_src_top_k_hits.json", "w") as f:
                json.dump(valid_multi_src_top_k_hits, f)

            if args.wandb:
                for src_idx in range(args.num_src):
                    cur_top_k_hit = valid_multi_src_top_k_hits[src_idx]
                    for k in range(args.top_k):
                        wandb.log(
                            {
                                f"valid_retriever_results/s_{src_idx}/top_{k}": cur_top_k_hit[
                                    k
                                ]
                            },
                            step=it,
                        )

            print("\n\n\n")

            # Test retrieval results per source using test questions, whether or not the source contains relevant information.
            print("\n\Generating Test Retrieval Results")

            test_q_info_lst = [
                corpus_to_q_a(d, is_valid=False) for d in multi_src_corpus
            ]
            test_multi_src_top_k_hits, total_retrieval_output_paths = (
                generate_multi_src_retrieval_results(
                    args=args,
                    model=contriever,
                    tokenizer=contriever_tokenizer,
                    multi_src_corpus=multi_src_corpus,
                    multi_src_embedding_files=multi_src_embedding_files,
                    log_name=dataset_log,
                    output_dir=args.retrieval_output_dir,
                    q_info_lst=test_q_info_lst,
                    src_reliabilities=src_reliabilities,
                    is_valid=False,
                    it=it,
                    top_k=args.top_k,
                )
            )

            with open(f"{dir_path}/test_multi_src_top_k_hits.json", "w") as f:
                json.dump(test_multi_src_top_k_hits, f)

            if args.wandb:
                for src_idx in range(args.num_src):
                    cur_top_k_hit = test_multi_src_top_k_hits[src_idx]
                    for k in range(args.top_k):
                        wandb.log(
                            {
                                f"test_retriever_results/s_{src_idx}/top_{k}": cur_top_k_hit[
                                    k
                                ]
                            },
                            step=it,
                        )

            print("\n\n\n")

            print("\n\Generating Baseline Retrieval Results")

            # Integrate corpus from all sources into a single source.
            integrated_corpus = integrate_multi_src_to_single_src(
                multi_src_corpus=multi_src_corpus
            )
            # Generate baseline embedding for the integrated corpus.
            baseline_embedding_files = generate_baseline_embedding(
                args=args,
                model=contriever,
                tokenizer=contriever_tokenizer,
                corpus=integrated_corpus,
                log_name=dataset_log,
                output_dir=args.retrieval_output_dir,
                it=it,
            )

            # Generate baseline retrieval results for validation questions that a single source contain relevant information.
            print("\n\Generating Baseline Retrieval Results for Validation Questions")
            valid_q_info = corpus_to_q_a(integrated_corpus, is_valid=True)
            base_valid_top_k_hits, _ = generate_baseline_retrieval_results(
                args=args,
                model=contriever,
                tokenizer=contriever_tokenizer,
                corpus=integrated_corpus,
                baseline_embedding_files=baseline_embedding_files,
                log_name=dataset_log,
                output_dir=args.retrieval_output_dir,
                q_info=valid_q_info,
                is_valid=True,
                it=it,
                top_k=10,
            )
            with open(f"{dir_path}/base_valid_top_k_hits.json", "w") as f:
                json.dump(base_valid_top_k_hits, f)

            print("\n\n\n")

            # Generate baseline retrieval results for test questions, whether or not the source contains relevant information.
            print("\n\Generating Baseline Retrieval Results for Test Questions")
            test_q_info = corpus_to_q_a(integrated_corpus, is_valid=False)
            base_test_top_k_hits, baseline_retrieval_output_path = (
                generate_baseline_retrieval_results(
                    args=args,
                    model=contriever,
                    tokenizer=contriever_tokenizer,
                    corpus=integrated_corpus,
                    baseline_embedding_files=baseline_embedding_files,
                    log_name=dataset_log,
                    output_dir=args.retrieval_output_dir,
                    q_info=test_q_info,
                    is_valid=False,
                    it=it,
                    top_k=10,
                )
            )
            with open(f"{dir_path}/base_test_top_k_hits.json", "w") as f:
                json.dump(base_test_top_k_hits, f)

            print("\n\n\n")

        # Generate reliability estimation dataset for the estimation phase.
        multi_src_est_rag_dataset = {}
        est_start_idx = 0
        est_end_idx = args.num_est_data
        for src_idx in range(args.num_src):
            retrieval_path = total_retrieval_output_paths[src_idx]
            multi_src_est_rag_dataset[f"s_{src_idx}"] = get_rag_dataset(
                retrieval_path=retrieval_path,
                top_k=args.top_k,
                prompt_type=args.prompt_type,
                start_idx=est_start_idx,
                end_idx=est_end_idx,
            )

        # Generate test dataset
        multi_src_test_rag_dataset = {}

        test_start_idx = args.num_est_data
        if args.num_test_data:
            test_end_idx = test_start_idx + args.num_test_data
        else:
            args.num_test_data = len(dataset)
            test_end_idx = len(dataset)
            print(f"num_test_data: {len(dataset) - args.num_est_data}")
            print()

        for src_idx in range(args.num_src):
            retrieval_path = total_retrieval_output_paths[src_idx]
            multi_src_test_rag_dataset[f"s_{src_idx}"] = get_rag_dataset(
                retrieval_path=retrieval_path,
                top_k=args.top_k,
                prompt_type=args.prompt_type,
                start_idx=test_start_idx,
                end_idx=test_end_idx,
            )

        print(f"{'='*60}")
        print("Weight Estimation Phase")
        print(f"{'='*60}\n")

        (
            estimated_weight_lst,
            est_reliabilities,
        ) = multi_source_weight_est(
            pipeline=pipeline,
            model_config=model_config,
            multi_src_est_rag_dataset=multi_src_est_rag_dataset,
            batch_size=args.batch_size,
            num_src=args.num_src,
            max_iter=args.max_iter,
            tol=args.tol,
            filtering_method=args.filtering_method,
            filtering_threshold=args.filtering_threshold,
            wandb=wandb if args.wandb else None,
            step=it,
            save_est_output=args.save_est_output,
            log_name=log_name if args.save_est_output else None,
            src_reliabilities=src_reliabilities,
            it=it,
        )

        if args.wandb:
            for src_idx, val in enumerate(est_reliabilities):
                wandb.log(
                    {f"estimation_phase/src_reliability/src_{src_idx}": val}, step=it
                )

        print(f"\nsampled src_reliabilities: {src_reliabilities}")
        print(f"est_reliabilities: {est_reliabilities}")

        print(f"{'='*60}")
        print("Test Phase")
        print(f"{'='*60}\n")

        # Using the results of MV for generating WMV results. This avoid redundancy of computation by reusing the results of MV which is already computed.
        if args.mv_to_wmv:
            # Inference results for MV
            total_outputs, total_gt, mv_em_results = multi_source_inference_mv(
                pipeline=pipeline,
                model_config=model_config,
                multi_src_rag_dataset=multi_src_test_rag_dataset,
                num_src=args.num_src,
                batch_size=args.batch_size,
                save_est_output=args.save_est_output,
                filtering_method=args.filtering_method,
                filtering_threshold=args.filtering_threshold,
                log_name=log_name if args.save_est_output else None,
                it=it,
                wandb=wandb if args.wandb else None,
            )
            total_mv_em_results.append(mv_em_results)

            # Inference results for WMV
            cur_wmv_em_results = [0 for _ in range(args.num_src + 1)]
            for i in range(1, args.num_src + 1):
                wmv_em_results = mv_to_wmv(
                    total_outputs=total_outputs,
                    total_gt=total_gt,
                    estimated_weight_lst=estimated_weight_lst,
                    top_k_src=i,
                )
                cur_wmv_em_results[i] = wmv_em_results

            total_wmv_em_results.append(cur_wmv_em_results)
            avg_wmv_em_results = np.mean(total_wmv_em_results, axis=0).tolist()

            # Inference results for Oracle Weighted MV (using true source reliabilities)
            oracle_weight = src_reliabilities

            cur_oracle_wmv_em_results = [0 for _ in range(args.num_src + 1)]
            for i in range(1, args.num_src + 1):
                wmv_em_results = mv_to_wmv(
                    total_outputs=total_outputs,
                    total_gt=total_gt,
                    estimated_weight_lst=oracle_weight,
                    top_k_src=i,
                )
                cur_oracle_wmv_em_results[i] = wmv_em_results

            total_oracle_wmv_em_results.append(cur_oracle_wmv_em_results)
            avg_oracle_wmv_em_results = np.mean(total_oracle_wmv_em_results, axis=0).tolist()

            print(f"mv_em_results: {mv_em_results}")
            print(f"avg_mv_em_results: {sum(total_mv_em_results)/(it+1)}")
            print()

            print(f"wmv_em_results: {cur_wmv_em_results}")
            print(f"avg_wmv_em_results: {avg_wmv_em_results}")
            print()

            print(f"oracle_wmv_em_results: {cur_oracle_wmv_em_results}")
            print(f"avg_oracle_wmv_em_results: {avg_oracle_wmv_em_results}")
            print()

            if args.wandb:
                wandb.log({f"MV/em_results": mv_em_results}, step=it)
                wandb.log(
                    {f"MV/avg_em_results": sum(total_mv_em_results) / (it + 1)}, step=it
                )

                for i in range(1, args.num_src + 1):
                    wandb.log(
                        {f"WMV/em_results/top_{i}_src": cur_wmv_em_results[i]},
                        step=it,
                    )
                    wandb.log(
                        {f"WMV/avg_em_results/top_{i}_src": avg_wmv_em_results[i]},
                        step=it,
                    )

                    wandb.log(
                        {f"ORACLE_WMV/em_results/top_{i}_src": cur_oracle_wmv_em_results[i]},
                        step=it,
                    )
                    wandb.log(
                        {f"ORACLE_WMV/avg_em_results/top_{i}_src": avg_oracle_wmv_em_results[i]},
                        step=it,
                    )

                wandb.log(
                    {f"Full WMV/em_results": cur_wmv_em_results[args.num_src]},
                    step=it,
                )
                wandb.log(
                    {f"Full WMV/avg_em_results": avg_wmv_em_results[args.num_src]},
                    step=it,
                )

                wandb.log(
                    {f"Full ORACLE_WMV/em_results": cur_oracle_wmv_em_results[args.num_src]},
                    step=it,
                )
                wandb.log(
                    {f"Full ORACLE_WMV/avg_em_results": avg_oracle_wmv_em_results[args.num_src]},
                    step=it,
                )
        else:
            wmv_em_results = reliablity_aware_inference(
                pipeline=pipeline,
                model_config=model_config,
                multi_src_retrieval_results=multi_src_test_rag_dataset,
                num_src=args.num_src,
                top_k_src=args.top_k_src,
                batch_size=args.batch_size,
                estimated_weight_lst=estimated_weight_lst,
            )
            total_wmv_em_results.append(wmv_em_results)

            print(f"wmv_em_results: {wmv_em_results}")
            print(f"total_wmv_em_results: {sum(total_wmv_em_results)/(it+1)}")

            if args.wandb:
                wandb.log({f"WMV/em_results/": wmv_em_results}, step=it)
                wandb.log(
                    {f"WMV/avg_em_results": sum(total_wmv_em_results) / (it + 1)},
                    step=it,
                )

        baseline_test_rag_dataset = get_rag_dataset(
            retrieval_path=baseline_retrieval_output_path,
            top_k=10,
            prompt_type=args.prompt_type,
            start_idx=test_start_idx,
            end_idx=test_end_idx,
        )

        baseline_em_results, baseline_target_em_results, baseline_outputs = (
            baseline_inference(
                pipeline=pipeline,
                model_config=model_config,
                multi_src_test_basline_retreival_results=baseline_test_rag_dataset,
                batch_size=args.batch_size,
                log_name=log_name,
                it=it,
            )
        )
        total_base_em_results.append(baseline_em_results)

        print(f"baseline_em_results: {baseline_em_results}")
        print(f"avg_baseline_em_results: {sum(total_base_em_results) / (it+1)}")
        print()

        if args.wandb:
            wandb.log({f"baseline/em_results": baseline_em_results}, step=it)
            wandb.log(
                {f"baseline/avg_baseline_em_results": sum(total_base_em_results) / (it + 1)},
                step=it,
            )


if __name__ == "__main__":
    main()
