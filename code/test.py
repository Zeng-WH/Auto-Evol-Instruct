import json
import codecs
import time
import openai
import random
import sys
import os
from pathlib import Path
import logging
import threading


# 从环境变量中获取 OpenAI API 配置
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_key = os.getenv('OPENAI_API_KEY')

from concurrent.futures import ThreadPoolExecutor

def get_oai_completion_message(messages):

    try: 
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages = messages,
            temperature=0.0,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0)
        
        res = response.choices[0].message["content"]    
        gpt_output = res
        return gpt_output
    except requests.exceptions.Timeout:
        # Handle the timeout error here
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.error.InvalidRequestError as e:
        # Handle the invalid request error here
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.error.APIError as e:
        if "The operation was timeout" in str(e):
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
#             time.sleep(3)
            return get_oai_completion_message(messages)            
        else:
            # Handle other API errors here
            print(f"The OpenAI API returned an error: {e}")
            return None
    except openai.error.RateLimitError as e:
        return get_oai_completion_message(messages)

def call_chatgpt_message(messages):
    success = False
    re_try_count = 20
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion_message(messages)
            
            if ans != "":
                success = True
        except:
            #time.sleep(10)
            print('retry for sample:', json.dumps(messages))
            print()
        time.sleep(10)
    return ans
def get_oai_completion_ori(prompt):

    try: 
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages = [{"role":"system","content":"You are an AI assistant that follows instruction extremely well. You need to provide the final answer in the format of '\n\nThe answer is: {Answer}.\n\n' at the end of the response, where {Answer} represents the numerical value of the answer."},
             {"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0)
        
        res = response.choices[0].message["content"]    
        gpt_output = res
        return gpt_output
    except requests.exceptions.Timeout:
        # Handle the timeout error here
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.error.InvalidRequestError as e:
        # Handle the invalid request error here
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.error.APIError as e:
        if "The operation was timeout" in str(e):
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
#             time.sleep(3)
            return get_oai_completion(prompt)            
        else:
            # Handle other API errors here
            print(f"The OpenAI API returned an error: {e}")
            return None
    except openai.error.RateLimitError as e:
        return get_oai_completion(prompt)

def call_chatgpt_ori(ins):
    success = False
    re_try_count = 20
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion_ori(ins)
            
            if ans != "":
                success = True
        except:
            #time.sleep(10)
            print('retry for sample:', ins)
            print()
        time.sleep(10)
    return ans


def get_oai_completion_meta(prompt):

    try: 
        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages = messages,
            temperature=0.0,
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0)
        
        res = response.choices[0].message["content"] 
        gpt_output = res
        return gpt_output
    except requests.exceptions.Timeout:
        # Handle the timeout error here
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.error.InvalidRequestError as e:
        # Handle the invalid request error here
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.error.APIError as e:
        if "The operation was timeout" in str(e):
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
#             time.sleep(3)
            return get_oai_completion(prompt)            
        else:
            # Handle other API errors here
            print(f"The OpenAI API returned an error: {e}")
            return None
    except openai.error.RateLimitError as e:
        return get_oai_completion(prompt)

def call_chatgpt_meta(ins):
    success = False
    re_try_count = 20
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion_meta(ins)
            
            if ans != "":
                success = True
        except:
            #time.sleep(10)
            print('retry for sample:', ins)
            print()
        time.sleep(10)
    return ans

import tqdm

all_json = []
# 定义一个线程锁，用来保证多个线程对all_json的操作是原子的
lock = threading.Lock()



def process_prompt(item):
    
    user_input = evol_prompt2 + item["raw"]["inst"]
    # resp = call_chatgpt_meta(user_input)
    temp_message = []
    temp_message.append({"role": "user", "content": user_input})
    resp = call_chatgpt_message(temp_message)    
    item["auto_evol"] = {}
    
    item["auto_evol"]["auto_detail"] = resp
    
    temp_text = resp.split("#Finally Rewritten Instruction#:")[-1]
    temp_text = temp_text.lstrip("\n")
    
    item["auto_evol"]["auto_inst"] = temp_text
    
    time.sleep(10)
    
    item["auto_evol"]["auto_output"] = call_chatgpt_ori(temp_text)
    
    time.sleep(5)
    

    
    all_json.append(item)
    
    

def process_data(data, num_threads):
    pbar = tqdm.tqdm(total=len(data))
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for item in data:
            executor.submit(process_prompt, item)
            pbar.update(1)
    pbar.close()
    
# with open(r"/workspaceblobstore/v-weihaozeng/auto_evol/math_data/GSM8K_train_7k.json", "r", encoding="utf-8") as r:
#     alpaca_data = json.load(r)
    
with open(r"/share/project/weihao/auto_evol/result/gsm8k_left3.json", "r") as r:
    #trn_lines = r.readlines()

    #trn_set = [json.loads(l) for l in trn_lines]
    trn_set = json.load(r)
alpaca_data = []

for item in trn_set:
    temp_json = {}
    temp_json["raw"] = {}
    temp_json["raw"]["inst"] = item["question"]
    temp_json["raw"]["output"] = item["answer"]
    alpaca_data.append(temp_json)
with open("/share/project/weihao/auto_evol/result/auto_evol_llama3_optimizer.json","r") as r:
    opti_data = json.load(r)
evol_prompt2 = opti_data[8]["evol_prompt"]    

process_data(alpaca_data, 50)



with open(r"/share/project/weihao/auto_evol/result/gsk8k_evol_7k_auto_v0.8.3_left_4.json", "w", encoding="utf-8") as w:
    
    json.dump(all_json, w)
    #alpaca_data = json.load(r)



