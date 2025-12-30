#!/bin/bash

p_rel=0.6
q_prior=0.6

num_est_data=200
num_test_data=0
exp_iter=10
eval_dataset=./dataset/NQ.json
retrieval_output_dir=./contriever_results

# Define log directory
log_dir=./logs/beta_prior
mkdir -p $log_dir

# Define GPU list
gpus=(4 5)                                                                                                                                                                                                                                                                                                                                                                                                                                         

# Queue of tasks to run. The code will run the tasks in the order of the num_srcs array parallelly when gpus are available.
num_srcs=(3 4 5 6 7 8 9) # num_src

# Array to store background job PIDs
declare -a pids=()
# Array to track which GPU each PID is using (ensures no GPU overlap)
declare -A pid_to_gpu=()

# Function to run a task on a specific GPU
run_task() {
    local num_src=$1
    local cuda_device=$2
    local log_file=$3
    local offset=$((num_src - 3))
    local master_port=$((30077 + offset))
    
    CUDA_VISIBLE_DEVICES=$cuda_device python -m torch.distributed.run --nproc_per_node 1 \
    --master_port=$master_port main.py \
    --p_rel $p_rel \
    --q_prior $q_prior \
    --num_src $num_src \
    --num_est_data $num_est_data \
    --num_test_data $num_test_data \
    --eval_dataset $eval_dataset \
    --retrieval_output_dir $retrieval_output_dir \
    --exp_iter $exp_iter \
    --mv_to_wmv \
    --save_est_output \
    --filtering_method align_score \
    --filtering_threshold 0.1 \
    --project_name RA_RAG_official_test \
    --wandb \
    > "$log_file" 2>&1
}

task_index=0
num_gpus=${#gpus[@]}

# Launch initial batch of tasks (up to number of GPUs)
for i in $(seq 0 $((num_gpus - 1))); do
    if [ $task_index -lt ${#num_srcs[@]} ]; then
        num_src=${num_srcs[$task_index]}
        gpu=${gpus[$i]}
        log_file="$log_dir/num_src_${num_src}_gpu_${gpu}_$(date +%Y%m%d_%H%M%S).log"
        
        # Run task in background
        run_task $num_src $gpu "$log_file" &
        pid=$!
        pids+=($pid)
        pid_to_gpu[$pid]=$gpu
        
        echo "Launched task num_src=$num_src on GPU $gpu (PID: $pid, log: $log_file)"
        ((task_index++))
    fi
done

# Process remaining tasks: when a task finishes, launch the next one
while [ $task_index -lt ${#num_srcs[@]} ]; do
    # Wait for any background job to finish
    wait -n
    
    # Find which PID(s) finished and remove them from the array
    # Check all PIDs to handle potential simultaneous completions
    finished_pids=()
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        if ! kill -0 $pid 2>/dev/null; then
            finished_pids+=($pid)
        fi
    done
    
    # Process finished PIDs (launch new tasks on freed GPUs)
    for finished_pid in "${finished_pids[@]}"; do
        freed_gpu=${pid_to_gpu[$finished_pid]}
        
        # Remove finished PID from tracking arrays
        for i in "${!pids[@]}"; do
            if [ "${pids[$i]}" == "$finished_pid" ]; then
                unset pids[$i]
                break
            fi
        done
        unset pid_to_gpu[$finished_pid]
        pids=("${pids[@]}")  # Reindex array
        
        echo "Task on GPU $freed_gpu finished (PID: $finished_pid)"
        
        # Launch next task on the freed GPU (if tasks remain)
        if [ $task_index -lt ${#num_srcs[@]} ]; then
            num_src=${num_srcs[$task_index]}
            log_file="$log_dir/num_src_${num_src}_gpu_${freed_gpu}_$(date +%Y%m%d_%H%M%S).log"
            run_task $num_src $freed_gpu "$log_file" &
            new_pid=$!
            pids+=($new_pid)
            pid_to_gpu[$new_pid]=$freed_gpu
            
            echo "Launched task num_src=$num_src on GPU $freed_gpu (PID: $new_pid, log: $log_file)"
            ((task_index++))
        fi
    done
done

# Wait for all remaining tasks to finish
echo "Waiting for all remaining tasks to finish..."
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All tasks completed!"

