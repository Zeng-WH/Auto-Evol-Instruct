# Auto Evol-Instruct

## Optimize Evolving Method

优化evolving method

需要指定以下脚本的参数完成优化

1. 配置openai的相关参数，包括api_type (azure etc), api_base, api_version以及api_key.  (其他LLM也支持，目前已经测试过Lllam3-70b)

2. 指定用于进化的模型 (evol_engine)，目前已经测试过的模型包括gpt-3.5, gpt-4以及Llama3-70b

3. 指定用于优化的模型 (evol_engine)，目前已经测试过的模型包括gpt-3.5, gpt-4以及Llama3-70b

4. 输入文件，由input_file指定，格式为字典，分别有"question", "answer"两个key

5. 输出文件，由output_file指定，包括优化好的evolving method

6. num_threads表示最大的并发数

7. batch_size表示batch size

8. dev_batch_size表示dev batch size

9. total_step表示优化总步数

10. sample_num表示Multiple Optimizations的次数

## Evol Instruction
指令进化代码（给定evolving method的情况下大规模进化Seed Dataset）

需要指定以下脚本的参数完成进化

1. 配置openai的相关参数，包括api_type (azure etc), api_base, api_version以及api_key.  (其他LLM也支持，目前已经测试过Lllam3-70b)

2. 指定用于进化的模型，目前已经测试过的模型包括gpt-3.5, gpt-4以及Llama3-70b

3. 输入文件，由input_file指定，格式为字典，分别有"question", "answer"两个key

4. evol_prompt_file表示通过Auto Evol-Instruct得到的一系列的evolving method, 按照evol_prompt_idx可以指定不同step的evovling method, 比如evol_prompt_idx=0表示initial evolving method

5. output_file表示输出的文件位置

6. num_threads表示最大的并发数


```bash
cd code

python evol_instruct.py \
    --input_file ../data/gsm8k_example.json \
    --evol_prompt_file ../data/autoevol_llama3_optimizer.json \
    --evol_prompt_idx 0 \
    --output_file ../data/output.json \
    --num_threads 50 \
    --api_type  \
    --evol_engine gpt-35-turbo \
    --api_base  \
    --api_version  \
    --api_key  \

```

