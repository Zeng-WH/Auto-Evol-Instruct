import json
import time
import openai
import sys
import os
import threading
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tqdm

def get_oai_completion_message(messages, evol_engine):
    try: 
        response = openai.ChatCompletion.create(
            engine=evol_engine,
            messages=messages,
            temperature=0.0,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        res = response.choices[0].message["content"]    
        return res
    except requests.exceptions.Timeout:
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.error.InvalidRequestError as e:
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.error.APIError as e:
        if "The operation was timeout" in str(e):
            print("The OpenAI API request timed out. Please try again later.")
            return get_oai_completion_message(messages, evol_engine)            
        else:
            print(f"The OpenAI API returned an error: {e}")
            return None
    except openai.error.RateLimitError as e:
        return get_oai_completion_message(messages, evol_engine)

def call_chatgpt_message(messages, evol_engine):
    success = False
    re_try_count = 20
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion_message(messages, evol_engine)
            if ans != "":
                success = True
        except:
            print('retry for sample:', json.dumps(messages))
        time.sleep(10)
    return ans

def get_oai_completion_ori(prompt, evol_engine):
    try: 
        response = openai.ChatCompletion.create(
            engine=evol_engine,
            messages=[
                #{"role": "system", "content": "You are an AI assistant..."},
                {"role":"system","content":"You are an AI assistant that follows instruction extremely well. You need to provide the final answer in the format of '\n\nThe answer is: {Answer}.\n\n' at the end of the response, where {Answer} represents the numerical value of the answer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        res = response.choices[0].message["content"]
        return res
    except requests.exceptions.Timeout:
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.error.InvalidRequestError as e:
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.error.APIError as e:
        if "The operation was timeout" in str(e):
            print("The OpenAI API request timed out. Please try again later.")
            return get_oai_completion_ori(prompt, evol_engine)            
        else:
            print(f"The OpenAI API returned an error: {e}")
            return None
    except openai.error.RateLimitError as e:
        return get_oai_completion_ori(prompt, evol_engine)

def call_chatgpt_ori(ins, evol_engine):
    success = False
    re_try_count = 20
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion_ori(ins, evol_engine)
            if ans != "":
                success = True
        except:
            print('retry for sample:', ins)
        time.sleep(10)
    return ans

def process_prompt(item, evol_prompt2, evol_engine):
    user_input = evol_prompt2 + item["raw"]["inst"]
    temp_message = [{"role": "user", "content": user_input}]
    resp = call_chatgpt_message(temp_message, evol_engine)
    item["auto_evol"] = {}
    item["auto_evol"]["auto_detail"] = resp
    temp_text = resp.split("#Finally Rewritten Instruction#:")[-1].lstrip("\n")
    item["auto_evol"]["auto_inst"] = temp_text
    time.sleep(10)
    item["auto_evol"]["auto_output"] = call_chatgpt_ori(temp_text, evol_engine)
    time.sleep(5)
    return item

# def process_data(data, num_threads, evol_prompt2, evol_engine):
#     all_json = []
#     pbar = tqdm.tqdm(total=len(data))
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         for item in data:
#             processed_item = executor.submit(process_prompt, item, evol_prompt2, evol_engine).result()
#             all_json.append(processed_item)
#             print(len(all_json))
#             print(all_json[-1])
#             pbar.update(1)
#     pbar.close()
#     return all_json

def process_data(data, num_threads, evol_prompt2, evol_engine):
    all_json = []
    pbar = tqdm.tqdm(total=len(data))
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for item in data:
            futures.append(executor.submit(process_prompt, item, evol_prompt2, evol_engine))
        
        for future in tqdm.tqdm(futures):
            processed_item = future.result()
            all_json.append(processed_item)
            pbar.update(1)
    
    pbar.close()
    return all_json


def main(args):
    # 从环境变量中获取 OpenAI API 配置
    openai.api_type = args.api_type
    openai.api_base = args.api_base
    openai.api_version = args.api_version
    openai.api_key = args.api_key

    # 读取输入文件
    with open(args.input_file, "r", encoding="utf-8") as r:
        trn_set = json.load(r)
    alpaca_data = [{"raw": {"inst": item["question"], "output": item["answer"]}} for item in trn_set]

    # 读取进化提示
    with open(args.evol_prompt_file, "r") as r:
        opti_data = json.load(r)
    evol_prompt2 = opti_data[args.evol_prompt_idx]["evol_prompt"]

    # 处理数据
    processed_data = process_data(alpaca_data, args.num_threads, evol_prompt2, args.evol_engine)

    # 写入输出文件
    with open(args.output_file, "w", encoding="utf-8") as w:
        json.dump(processed_data, w)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data using OpenAI API.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--evol_prompt_file", type=str, required=True, help="Path to the evolution prompt JSON file.")
    parser.add_argument("--evol_engine", type=str, required=True, help="Evol Engine")
    parser.add_argument("--evol_prompt_idx", type=int, default=0, help="Evol Prompt Use to Evol Instruct")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--num_threads", type=int, default=50, help="Number of threads to use for processing.")
    parser.add_argument("--api_type", type=str, default=os.getenv('OPENAI_API_TYPE'), help="OpenAI API type.")
    parser.add_argument("--api_base", type=str, default=os.getenv('OPENAI_API_BASE'), help="OpenAI API base.")
    parser.add_argument("--api_version", type=str, default=os.getenv('OPENAI_API_VERSION'), help="OpenAI API version.")
    parser.add_argument("--api_key", type=str, default=os.getenv('OPENAI_API_KEY'), help="OpenAI API key.")
    
    args = parser.parse_args()
    main(args)
