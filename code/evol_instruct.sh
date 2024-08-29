python evol_instruct.py \
    --input_file ../data/gsm8k_example.json \
    --evol_prompt_file ../data/autoevol_llama3_optimizer.json \
    --evol_prompt_idx 0 \
    --output_file ../data/output.json \
    --num_threads 50 \
    --api_type azure \
    --evol_engine gpt-35-turbo \
    --api_base https://baaisolution-ae.openai.azure.com/ \
    --api_version 2023-03-15-preview \
    --api_key 2d196d16e1bc470c86689efdc7ca4943 \


