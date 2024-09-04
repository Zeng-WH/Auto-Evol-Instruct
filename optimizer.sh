python optimizer.py \
    --input_file /share/project/weihao/auto_evol/result/gsm8k_left1.json \
    --output_file ../data/evol_prompt.json \
    --api_type azure \
    --evol_engine gpt-35-turbo \
    --optimizer_engine gpt-35-turbo \
    --batch_size 10 \
    --dev_batch_size 50 \
    --total_step 10 \
    --sample_num 5 \
    --api_base  \
    --api_version  \
    --api_key  \
    --num_threads 50 \

