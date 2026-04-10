from openai import OpenAI
import json
import re
import random
import time
import requests
import os
moda_api=''
api_list=[moda_api]
client_gpt = OpenAI(api_key="",base_url='')

eval_models={"gpt-4o-mini":'gpt-4o-mini',
             "gpt-4":'gpt-4',
             "gpt-4o":'gpt-4o',
             "gpt-3.5-turbo":'gpt-3.5-turbo',
             "llama_8B":'llama-3.1-8b-instruct',
             "llama_70B":'meta-llama/Llama-3.1-70B-Instruct',
             "llama_405B":'meta-llama/Llama-3.1-405B-Instruct',
             "llama-8B":"LLM-Research/Meta-Llama-3.1-8B-Instruct",
             "llama-70B":"llama-3.1-70b",
             "llama-405B":"LLM-Research/Meta-Llama-3.1-405B-Instruct",
             "deepseek-v3":"deepseek-ai/DeepSeek-V3",
             "deepseek-r1":"deepseek-ai/DeepSeek-R1",
             "Qwen3":"Qwen/Qwen3-8B",
             "Qwen3-30B":"Qwen/Qwen3-30B-A3B",
             "Qwen2.5":"Qwen/Qwen2.5-7B-Instruct"
}

client_ds=OpenAI(api_key="",base_url='')
client_llama=OpenAI(api_key="",base_url="")


def chat_completion(model_name, message,temperature=0.2):
    fail_turn=3
    turn=0
    
    while True:
        turn+=1
        if turn>fail_turn:
            return ''
        try:
            res=''
            if model_name=='llama-70B':
                res=getfromllama(message,eval_models[model_name],temperature=temperature)
            elif "deepseek" in model_name or model_name=='llama-405B':
                res=getfromModa(message,eval_models[model_name],temperature=temperature)
            elif model_name=='llama_8B':
                res=getfromllama(message,eval_models[model_name],temperature=temperature)
            else:
                res=getfromOpenAI(message,eval_models[model_name],temperature=temperature)
            return res
        except Exception as e:
            #如果出事就等10分钟
            print(e)
            print('wait for 5mins')
            time.sleep(300)
    
def getfromdeepseek(message,model,temperature):
    completion = client_ds.chat.completions.create(
        model=model,
        messages=message,
        max_tokens=512,
        temperature=temperature,
        stream=False
    )
        
    return completion.choices[0].message.content


def getfromModa(message,model,temperature,):
    client_moda=OpenAI(api_key=random.choice(api_list), base_url="")
    completion = client_moda.chat.completions.create(
        model=model,
        messages=message,
        max_tokens=512,
        temperature=temperature,
        stream=False
    )
    time.sleep(0)
        
    return completion.choices[0].message.content

def getfromllama(message,model,temperature):
    completion = client_llama.chat.completions.create(
        model=model,
        messages=message,
        max_tokens=512,
        temperature=temperature,
        stream=False
    )

    return completion.choices[0].message.content

def getfromOpenAI(message,model='gpt-4o-mini',temperature=0.8):
  completion = client_gpt.chat.completions.create(
    model=model,
    messages=message,
    max_tokens=512,
    temperature=temperature,
  )
 
  return completion.choices[0].message.content



def load_data(input_path):
    dataset = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dataset.append(json.loads(line))
    return dataset


def check_answer_bool(bot_answer) -> int:
    # 将bot_answer转换为小写，并使用正则表达式将其拆分为单词
    ans = re.split(r"\W+", bot_answer.lower())
    # 如果单词列表中包含'yes'，则返回1
    if 'yes' in ans:
        return 1
    # 如果单词列表中包含'no'，则返回0
    elif 'no' in ans:
        return 0
    # 如果单词列表中既不包含'yes'，也不包含'no'，则返回-1
    else:
        return -1


def check_answer_mc(output, mc2_targets):
    match = re.search(r'Option\s*(\d+)', output, re.IGNORECASE | re.DOTALL)
    if not match: # then match literature parenthesis
         match = re.search(r'\((\d+)\)', output, re.IGNORECASE | re.DOTALL) 
    if match:
        choice = int(match.group(1)) - 1
        if choice == len(mc2_targets):  # don't know, we treat it not change belief
            return -1
        if choice > len(mc2_targets): # all options are incorrect, we treat is change belief
            return 0
        if mc2_targets[choice]['score'] == 1:
            return 1
    else: 
        assert(0)

    return 0


def check_answer_em(output, mc2_targets):
    check = False
    for choice in mc2_targets:
        if choice['text'].lower().strip().strip('.') in output.lower().strip().strip('.'):
            return True
    if "don't know" in output.lower().strip().strip('.'):
        return True
    return check


def replace_variables(text, variables):
    for variable, value in variables.items():
        text = text.replace(variable, str(value))
    return text


def extract_prompt(file_path, variables):
    with open(file_path, 'r') as file:
        content = file.read()

    #分别是prompt中system和user的起始标记
    system_start_marker = "==== SYSTEM ===="
    user_start_marker = "==== USER ===="

    system_start = content.find(system_start_marker) + len(system_start_marker)
    system_end = content.find(user_start_marker)
    system_text = content[system_start:system_end].strip()

    user_start = content.find(user_start_marker) + len(user_start_marker)
    user_text = content[user_start:].strip()

    system_text = replace_variables(system_text, variables)
    user_text = replace_variables(user_text, variables)

    return system_text, user_text


def get_variables(data, fallacy):
    return{
        "<QUESTION>":data['question'],
        "<FALLACY>":fallacy,
        "<CTRL>":data['adv']['control'],
        "<ANSWER>":data['adv']['target']
    }

def getDictFromJsonl(file_path):
    if os.path.exists(file_path)==False:
        return []
    data=[]
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r',encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    elif file_path.endswith('.json'):
        with open(file_path, 'r',encoding='utf-8') as f:
            data = json.load(f)
    
    return data

def safe_json_loads(json_str):
    try:
        # 尝试解析JSON字符串
        if isinstance(json_str, str):
            pattern = r'\{.*?\}'

            match = re.search(pattern, json_str, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else: 
                return None
        else:
            print("输入不是字符串")
            return None
        
    except json.JSONDecodeError:
        # 如果解析失败，打印错误信息并返回None
        print("JSON解析失败，将尝试重新获取数据。")
        return None
    

def append_to_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

            
def record(log_file, log):
    with open(log_file, 'a') as f:
        #先记录当前时间
        f.write('==================================\n')
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        f.write(log + '\n')

def random_id():
    #随机生成包括字母和数字的字符串
    return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))


import logging
import sys
class Logger:
    def __init__(self, log_dir=None,log_file=None ,enabled=True, pad_length=50):
        if os.path.exists(log_dir) == False:
            os.makedirs(log_dir)
            
        self._logger = self._get_logger(os.path.join(log_dir, log_file)) if enabled else None
        self._pad_length = pad_length
        self._log_file_path= os.path.join(log_dir, log_file) if enabled else None

    def _pad_message(self, message):
        return (" " + message + " ").center(self._pad_length, '=')

    def info(self, message, pad=False):
        if self._logger is not None:
            #print(message)
            message = self._pad_message(message) if pad else message
            self._logger.info(message)

    def raw(self, message=""):
        """
        Write a raw line (without any formatter, e.g., time or level).
        """
        if self._log_file_path:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def line(self):
        if self._logger is not None:
            self._logger.info('=' * self._pad_length)

    @staticmethod
    def _get_logger(log_dir=None):
        logger = logging.getLogger("MyCustomLogger")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # 不传到 root logger，避免其他库的 DEBUG 输出污染
        if logger.handlers:
            return logger  # 防止重复添加 handler

        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        if log_dir is not None:
            file_handler = logging.FileHandler(log_dir,encoding='utf-8')
            # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            # file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger
    
def get_generate_prompt(prefix,fallacy):
    if fallacy=='Hasty Generalization':
        return prefix + 'HG.txt'
    elif fallacy=='Ad Hominem':
        return prefix + 'AH.txt'
    elif fallacy=='Straw Man':
        return prefix + 'SM.txt'
    elif fallacy=='Red Herring':
        return prefix + 'RH.txt'
    elif fallacy=='False Causality':
        return prefix + 'FC.txt'
    elif fallacy=='False Dilemma':
        return prefix + 'FD.txt'
    elif fallacy=='Slippery Slope':
        return prefix + 'SS.txt'
    elif fallacy=='Appeal to Authority':
        return prefix + 'AA.txt'
    elif fallacy=='Equivocation':
        return prefix + 'E.txt'
    elif fallacy=='Circular Reasoning':
        return prefix + 'CR.txt'
    
def hebin(json_file1,json_file2,output_file):
    data1=getDictFromJsonl(json_file1)
    data2=getDictFromJsonl(json_file2)
    if len(data1)!=len(data2):
        print('长度不一致')
        return
    data=[]
    for index,item in enumerate(data1):
        item['adv'].update(data2[index]['adv'])
        data.append(item)

    append_to_jsonl(output_file,data)
        


