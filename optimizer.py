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
from concurrent.futures import ThreadPoolExecutor
import tqdm
import re
import numpy as np

import argparse

def is_false_string(input_string):
    # 使用字符串的startswith方法检查是否以"Understood"开头，并且使用endswith方法检查是否以"?"结尾
    if input_string.startswith("Understood") and input_string.endswith("?"):
        return False

    elif input_string.startswith("What") and input_string.endswith("?"):
        return False
    elif input_string.startswith("That is correct") and input_string.endswith("?"):
        return False   
    elif input_string.startswith("Thank you") and input_string.endswith("?"):
        return False   
    elif input_string.startswith("Sure") and input_string.endswith("?"):
        return False

    elif input_string.startswith("Great") and input_string.endswith("?"):
        return False
    elif input_string.startswith("I'm sorry") and input_string.endswith("?"):
        return False

    elif input_string.startswith("Is there anything else") and input_string.endswith("?"):
        return False

    elif input_string.startswith("As an AI assistant") and input_string.endswith("?"):
        return False

    elif input_string.startswith("I understand") and input_string.endswith("?"):
        return False
    elif "please provide" in input_string.lower():
        return False
    elif "not possible" in input_string.lower():
        return False
    else:
        return True


def is_false_string(input_string):
    # 使用字符串的startswith方法检查是否以"Understood"开头，并且使用endswith方法检查是否以"?"结尾
    if input_string.startswith("Understood") and input_string.endswith("?"):
        return False

    elif input_string.startswith("What") and input_string.endswith("?"):
        return False
    elif input_string.startswith("That is correct") and input_string.endswith("?"):
        return False   
    elif input_string.startswith("Thank you") and input_string.endswith("?"):
        return False   
    elif input_string.startswith("Sure") and input_string.endswith("?"):
        return False

    elif input_string.startswith("Great") and input_string.endswith("?"):
        return False
    elif input_string.startswith("I'm sorry") and input_string.endswith("?"):
        return False

    elif input_string.startswith("Is there anything else") and input_string.endswith("?"):
        return False

    elif input_string.startswith("As an AI assistant") and input_string.endswith("?"):
        return False

    elif input_string.startswith("I understand") and input_string.endswith("?"):
        return False
    elif "please provide" in input_string.lower():
        return False
    elif "not possible" in input_string.lower():
        return False
    else:
        return True
    
def cal_success(evol_list):
    clean_list = []
    
    
    for item in evol_list:
        if is_false_string(item) == True:
            clean_list.append(item)
            
    return len(clean_list) / len(evol_list)


def cal_avglen(evol_list):
    import numpy as np
    len_list = []
    for item in evol_list:
        len_list.append(len(item))
        
    return np.mean(len_list)
        
        
def calevol_metrics(evol_list):
    inst_list = []
    
    resp_list = []
    
    len_list = []
    
    for item in evol_list:
        inst_list.append(item["input"])
        resp_list.append(item["output"])
        
        len_list.append(cal_avglen(resp_list) / cal_avglen(inst_list))
        
        
    return cal_success(resp_list), cal_success(resp_list)*(cal_avglen(resp_list) / cal_avglen(inst_list))


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



def get_oai_completion_message_gpt4(messages, optimizer_engine):
    try: 
        response = openai.ChatCompletion.create(
            engine=optimizer_engine,
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
            return get_oai_completion_message_gpt4(messages, optimizer_engine)            
        else:
            print(f"The OpenAI API returned an error: {e}")
            return None
    except openai.error.RateLimitError as e:
        return get_oai_completion_message_gpt4(messages, optimizer_engine)

def call_chatgpt_message_gpt4(messages, optimizer_engine):
    success = False
    re_try_count = 20
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion_message_gpt4(messages, optimizer_engine)
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

                {"role":"system","content":"You are an AI assistant that follows instruction extremely well."},
                #{"role":"system","content":"You are an AI assistant that follows instruction extremely well. You need to provide the final answer in the format of '\n\nThe answer is: {Answer}.\n\n' at the end of the response, where {Answer} represents the numerical value of the answer."},
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

def trn_nstage(input_list, n, evol_prompt):
    result_list = []
    def trn_nstage_item(index, item):
        temp_json = {}
        temp_json["id"] = index
        init_inst = item["raw"]["inst"].replace("\nInput:", "\r\n")
        temp_json["stage0"] = init_inst
        temp_inst = init_inst
        for i in range(n):
            temp_message = []
            temp_message.append({"role": "user", "content": evol_prompt + temp_inst})
            temp_inst1 = call_chatgpt_message(temp_message, evol_engine)
            if "#Finally Rewritten Instruction#:" in temp_inst1:
                temp_inst = temp_inst1.split("#Finally Rewritten Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")
                
            elif "#Final Rewritten Instruction#:" in temp_inst1:
                temp_inst = temp_inst1.split("#Final Rewritten Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")                

            elif "#Final Evolved Instruction#:" in temp_inst1:
                temp_inst = temp_inst1.split("#Final Evolved Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")
            elif "#Final Refined Instruction#:" in temp_inst1:
                temp_inst = temp_inst1.split("#Final Refined Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")
            elif "Final Rewritten Instruction:" in temp_inst1:
                temp_inst = temp_inst1.split("Final Rewritten Instruction:")[-1]
                temp_inst = temp_inst.lstrip("\n")   
            elif "Finally Rewritten Instruction:" in temp_inst1:
                temp_inst = temp_inst1.split("Finally Rewritten Instruction:")[-1]
                temp_inst = temp_inst.lstrip("\n")            
            else:
                temp_inst = temp_inst1.split("#Finally Rewritten Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")              
            # temp_inst = temp_inst1.split("#Finally Rewritten Instruction#:")[-1]
            # temp_inst = temp_inst.lstrip("\n")
            # temp_json["stage"+str(i+1)] = {}
            temp_inst = temp_inst.strip()
            temp_json["stage"+str(i+1)] = temp_inst
            # resp_message = []
            # resp_message.append({"role": "user", "content": temp_inst})
            
            # temp_resp1 = call_chatgpt_message(resp_message)
            # temp_json["stage"+str(i+1)]["resp"] = temp_resp1
            
            
            
            
        result_list.append(temp_json)
        
    def process_data(data, num_threads):
        pbar = tqdm.tqdm(total=len(data))
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for index, item in enumerate(data):
                executor.submit(trn_nstage_item, index, item)
                pbar.update(1)
        pbar.close()
        
    process_data(input_list, num_threads)
    
    
    sorted_list = sorted(result_list, key=lambda x: x['id'])      
    
    
    return sorted_list

def extract_content(markdown):
    # 定义要提取的正则表达式
    regex = r"```Optimized Method([\s\S]*?)```"
    # 使用re.search方法在字符串中搜索正则表达式
    match = re.search(regex, markdown)
    if match:
        method = match.group(1)
        return method
    #print(match.group(1))
    else:
        return ""


def opti_model(tool_prompt, trn_output):
    user_prompt1 = loss_prompt + json.dumps(trn_output)
    messgae_loss = []
    
    messgae_loss.append({"role": "user", "content": user_prompt1})
    trn_loss = call_chatgpt_message_gpt4(messgae_loss, optimizer_engine)
    
    if trn_loss == "":
        return tool_prompt
    messgae_loss.append({"role": "assistant", "content": trn_loss})
    
    user_prompt2 = optim_prompt + "```Method\n"+ tool_prompt + "```\n"
    
    messgae_loss.append({"role": "user", "content": user_prompt2})
    
    time.sleep(30)
    optim_output = call_chatgpt_message_gpt4(messgae_loss, optimizer_engine)

    if "```Optimized Method" in optim_output:
    
        temp_tool_prompt = extract_content(optim_output)
    
        if temp_tool_prompt == "":
            return tool_prompt
        else:
            return temp_tool_prompt
        
    else:
        return tool_prompt

def prompt_metric(input_list, n, evol_prompt):
    result_list = []
    def prompt_metric_item(index, item):
        temp_json = {}
        temp_json["id"] = index
        init_inst = item["raw"]["inst"].replace("\nInput:", "\r\n")
        temp_json["stage0"] = init_inst
        temp_inst = init_inst
        for i in range(n):
            temp_message = []
            temp_message.append({"role": "user", "content": evol_prompt + temp_inst})
            temp_inst1 = call_chatgpt_message(temp_message, evol_engine)
            if "#Finally Rewritten Instruction#:" in temp_inst1:
                temp_inst = temp_inst1.split("#Finally Rewritten Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")
                
            elif "#Final Rewritten Instruction#:" in temp_inst1:
                temp_inst = temp_inst1.split("#Final Rewritten Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")                

            elif "#Final Evolved Instruction#:" in temp_inst1:
                temp_inst = temp_inst1.split("#Final Evolved Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")
            elif "#Final Refined Instruction#:" in temp_inst1:
                temp_inst = temp_inst1.split("#Final Refined Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")
            # Final Rewritten Instruction:
            elif "Final Rewritten Instruction:" in temp_inst1:
                temp_inst = temp_inst1.split("Final Rewritten Instruction:")[-1]
                temp_inst = temp_inst.lstrip("\n")
            elif "Finally Rewritten Instruction:" in temp_inst1:
                temp_inst = temp_inst1.split("Finally Rewritten Instruction:")[-1]
                temp_inst = temp_inst.lstrip("\n")                
            else:
                temp_inst = temp_inst1.split("#Finally Rewritten Instruction#:")[-1]
                temp_inst = temp_inst.lstrip("\n")            
            # temp_inst = temp_inst1.split("#Finally Rewritten Instruction#:")[-1]
            # temp_inst = temp_inst.lstrip("\n")

            temp_inst = temp_inst.strip()
            temp_json["stage"+str(i+1)] = {}
            temp_json["stage"+str(i+1)]["inst"] = temp_inst
            resp_message = []
            resp_message.append({"role": "user", "content": temp_inst})
            
            #temp_resp1 = call_chatgpt_message(resp_message)
            temp_resp1 = call_chatgpt_ori(temp_inst, evol_engine)
            temp_json["stage"+str(i+1)]["resp"] = temp_resp1
            
            
            
            
        result_list.append(temp_json)
        
    def process_data(data, num_threads):
        pbar = tqdm.tqdm(total=len(data))
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for index, item in enumerate(data):
                executor.submit(prompt_metric_item, index, item)
                pbar.update(1)
        pbar.close()
        
    process_data(input_list, num_threads)
    
    
    sorted_list = sorted(result_list, key=lambda x: x['id'])        

        
    all_score_list = []
    
    for index in range(n):
        temp_list = []
        
        for item in sorted_list:
            
            temp_list.append({"input": item["stage"+str(index+1)]["inst"], "output": item["stage"+str(index+1)]["resp"]})
        all_score_list.append(calevol_metrics(temp_list))

    return all_score_list, sorted_list        



init_prompt = '''
You are an Instruction Rewriter that rewrites the given #Instruction# into a more complex version.

Please follow the steps below to rewrite the given "#Instruction#" into a more complex version.

Step 1: Please read the "#Instruction#" carefully and list all the possible methods to make this instruction more complex (to make it a bit harder for well-known AI assistants such as ChatGPT and GPT4 to handle). Please do not provide methods to change the language of the instruction!

Step 2: Please create a comprehensive plan based on the #Methods List# generated in Step 1 to make the #Instruction# more complex. The plan should include serval methods from the #Methods List#.

Step 3: Please execute the plan step by step and provide the #Rewritten Instruction#. #Rewritten Instruction# can only add 10 to 20 words into the "#Instruction#".

Step 4: Please carefully review the #Rewritten Instruction# and identify any unreasonable parts. Ensure that the #Rewritten Instruction# is only a more complex version of the #Instruction#. Just provide the #Finally Rewritten Instruction# without any explanation.

Please reply strictly in the following format:
Step 1
#Methods List#: 
Step 2
#Plan#: 
Step 3
#Rewritten Instruction#:
Step 4
#Finally Rewritten Instruction#:


#Instruction#:
'''

loss_prompt = '''
The following list shows cases where an Instruction evolves into a more complex version of an Instruction. For each case, stage 0 represents the Instruction in its initial state, and each subsequent stage requires an increase in complexity based on the previous stage. Please identify cases that failed to evolve, and provide their case ID and reasons.
'''


optim_prompt = '''
I will provide you with the method for evolving the above instructions. You need to optimize this method based on the feedback from the evolution failure case, without harming the performance on other cases, and ensure that the complexity increase brought by the optimized method is not lower than the previous method. Please provide the optimized method in the following format. \n ```Optimized Method\n<Optimized Method Here>\n```
'''


def main(args):
    # 从环境变量中获取 OpenAI API 配置
    openai.api_type = args.api_type
    openai.api_base = args.api_base
    openai.api_version = args.api_version
    openai.api_key = args.api_key

    global evol_engine

    evol_engine = args.evol_engine
    global optimizer_engine

    global num_threads
    num_threads = args.num_threads


    optimizer_engine = args.optimizer_engine

    # 读取输入文件
    with open(args.input_file, "r", encoding="utf-8") as r:
        trn_set = json.load(r)
    alpaca_data = [{"raw": {"inst": item["question"], "output": item["answer"]}} for item in trn_set]

    trn_set = alpaca_data

    import random

    dev_set = trn_set.copy()

    random.shuffle(trn_set)
    random.shuffle(dev_set)

    bsz = args.batch_size

    dev_bsz = args.dev_batch_size

    trn_set = [trn_set[i:i+bsz] for i in range(0, len(trn_set), bsz)]
    dev_set = [dev_set[i:i+dev_bsz] for i in range(0, len(dev_set), dev_bsz)]

    # import pdb
    # pdb.set_trace()


    temp_prompt = init_prompt

    log_list = []

    for index, input_ls in enumerate(trn_set):
        if index == args.total_step:
            break

        print("----------Strep "+ str(index) + "----------")

        temp_list, sample_list = prompt_metric(dev_set[index], 1, temp_prompt)

        temp_success = [t[0] for t in temp_list]
        temp_success = np.mean(temp_success)
        temp_score = [t[1] for t in temp_list]
        temp_score = np.mean(temp_score)


        temp_log = {}
        temp_log["evol_prompt"] = temp_prompt
        trn_output = trn_nstage(input_ls, 1, temp_prompt)
        temp_log["evol_trajectory"] = trn_output
        log_list.append(temp_log)


        with open(args.output_file, "w") as w:
            json.dump(log_list, w)

        all_list = []

        for i in range(args.sample_num):
            temp_prompt1 = opti_model(temp_prompt, trn_output)
            temp_list1, sample_list1 = prompt_metric(dev_set[index], 1, temp_prompt1)
            temp_success1 = [t[0] for t in temp_list1]
            temp_success1 = np.mean(temp_success1)

            temp_score1 = [t[1] for t in temp_list1]
            temp_score1 = np.mean(temp_score1) 

            all_list.append({"evol_prompt": temp_prompt1, "score_list": temp_list1, "t_success":temp_success1 , "t_score": temp_score1, "sample_list": sample_list1})
        all_list = sorted(all_list, key=lambda x: x['t_score'], reverse=True)

        for i in all_list:
            if i["t_success"] >= temp_success and i["t_score"] >= temp_score:
                temp_prompt = i["evol_prompt"]
                temp_score = i["t_score"]
                temp_success = i["t_success"]
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data using OpenAI API.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--evol_engine", type=str, required=True, help="Evol Engine")
    parser.add_argument("--optimizer_engine", type=str, required=True, help="Optimizer Engine")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")
    parser.add_argument("--dev_batch_size", type=int, default=10, help="Dev Batch Size")
    parser.add_argument("--total_step", type=int, default=10, help="Total Steps")
    parser.add_argument("--sample_num", type=int, default=10, help="Sample Number")
    parser.add_argument("--api_type", type=str, default=os.getenv('OPENAI_API_TYPE'), help="OpenAI API type.")
    parser.add_argument("--api_base", type=str, default=os.getenv('OPENAI_API_BASE'), help="OpenAI API base.")
    parser.add_argument("--api_version", type=str, default=os.getenv('OPENAI_API_VERSION'), help="OpenAI API version.")
    parser.add_argument("--api_key", type=str, default=os.getenv('OPENAI_API_KEY'), help="OpenAI API key.")
    parser.add_argument("--num_threads", type=int, default=50, help="Number of threads to use for processing.")
    
    args = parser.parse_args()
    main(args)
        





















    

