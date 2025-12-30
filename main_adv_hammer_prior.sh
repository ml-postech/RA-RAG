#!/bin/bash

num_est_data=200
num_test_data=0
exp_iter=10
# eval_dataset=./dataset/NQ.json
# eval_dataset=./dataset/TQA.json
eval_dataset=./dataset/HotpotQA.json
retrieval_output_dir=./contriever_results


# Define log directory
log_dir=./logs/adv_hammer
mkdir -p $log_dir

# Define GPU list
gpus=(3 4 5)

# Queue of tasks to run. The code will run the tasks in the order of the arrays parallelly when gpus are available.
num_srcs=(9) # num_src
num_advs=(1 2 3) # num_adv

# Create task queue with all combinations of num_srcs and num_advs
declare -a tasks=()
for num_src in "${num_srcs[@]}"; do

    for num_adv in "${num_advs[@]}"; do
        tasks+=("${num_src}:${num_adv}")
    done
done

# Array to store background job PIDs
declare -a pids=()
# Array to track which GPU each PID is using (ensures no GPU overlap)
declare -A pid_to_gpu=()

# Function to run a task on a specific GPU
run_task() {
    local num_src=$1
    local num_adv=$2
    local cuda_device=$3
    local log_file=$4
    # Calculate unique port: base_port + (num_src - 3) * 100 + num_adv
    # This ensures unique ports for different combinations of num_src and num_adv
    local offset=$(( (num_src - 3) * 100 + num_adv ))
    local master_port=$((30077 + offset))
    
    CUDA_VISIBLE_DEVICES=$cuda_device python -m torch.distributed.run --nproc_per_node 1 \
    --master_port=$master_port main.py \
    --adv_hammer \
    --num_src $num_src \
    --num_adv $num_adv \
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
    if [ $task_index -lt ${#tasks[@]} ]; then
        task=${tasks[$task_index]}
        IFS=':' read -r num_src num_adv <<< "$task"
        IFS=$' \t\n'  # Reset IFS to default
        gpu=${gpus[$i]}
        log_file="$log_dir/num_src_${num_src}_num_adv_${num_adv}_gpu_${gpu}_$(date +%Y%m%d_%H%M%S).log"
        
        # Run task in background
        run_task $num_src $num_adv $gpu "$log_file" &
        pid=$!
        pids+=($pid)
        pid_to_gpu[$pid]=$gpu
        
        echo "Launched task num_src=$num_src num_adv=$num_adv on GPU $gpu (PID: $pid, log: $log_file)"
        ((task_index++))
    fi
done

# Process remaining tasks: when a task finishes, launch the next one
while [ $task_index -lt ${#tasks[@]} ]; do
    # Wait for any background job to finish
    # Check if there are any background jobs before waiting
    if [ ${#pids[@]} -eq 0 ]; then
        break
    fi
    wait -n || true  # Continue even if wait fails (e.g., no jobs)
    
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
        if [ $task_index -lt ${#tasks[@]} ]; then
            task=${tasks[$task_index]}
            IFS=':' read -r num_src num_adv <<< "$task"
            IFS=$' \t\n'  # Reset IFS to default
            log_file="$log_dir/num_src_${num_src}_num_adv_${num_adv}_gpu_${freed_gpu}_$(date +%Y%m%d_%H%M%S).log"
            run_task $num_src $num_adv $freed_gpu "$log_file" &
            new_pid=$!
            pids+=($new_pid)
            pid_to_gpu[$new_pid]=$freed_gpu
            
            echo "Launched task num_src=$num_src num_adv=$num_adv on GPU $freed_gpu (PID: $new_pid, log: $log_file)"
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

