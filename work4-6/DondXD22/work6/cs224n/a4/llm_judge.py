import re
import json
import os
from typing import List, Dict

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model

from gsm8k import print_response
import random

import time


# You may find these constants useful for structuring the judge's output.
MODEL_E_PREFERED_TAG = "<MODEL_E_BETTER>"
MODEL_F_PREFERED_TAG = "<MODEL_F_BETTER>"
NO_PREFERENCE_FOUND_TAG = "<NO_PREFERENCE_FOUND>"
USE_CHINESE='\n使用中文进行回答'


def load_alpaca_data() -> List[Dict[str, str]]:

    dataset = []
    with open("./data/alpaca_eval_first_30.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            dataset.append(example)

    return dataset

def llm_judge_template(query: str, response_E: str, response_F: str) -> str:
    """    
    构建一个用于LLM评判器评估两个模型回复的提示词。
    参数：
        query：向两个模型提出的问题（来自AlpacaEval）
        response_E：模型E对query的输出
        response_F：模型F对query的输出
    返回值：
        供LLM评判器使用的提示词。
        考虑：评判器是一个会输出自由格式文本的LLM。
        你将如何设计提示词，以便能够可靠地确定它更偏好哪个回复？
        你的llm_judge_template和extract_llm_judge_preference
        应协同工作。
    """
    # TODO complete for question 3b
    prompt=f"""
这里是来自两个不同模型关于同一个问题的回答，请判断哪一个模型的回复更好
问题：{query}
模型E回答：{response_E}
模型F回答：{response_F}
请在最后单列一行，使用以下标记来输出你的判断：
模型E更好：{MODEL_E_PREFERED_TAG}
模型F更好：{MODEL_F_PREFERED_TAG}
二者相同：{NO_PREFERENCE_FOUND_TAG}
    """
    return prompt


def extract_llm_judge_preference(judge_output: str) -> str:
    """
    从法官的输出中提取其偏好。

    参数：
        judge_output：从LLM法官采样的字符串。
    返回：
        一个字符串，表示法官偏好的回复。
    
    此函数应与你设计的llm_judge_template协同工作。
    如果法官的输出格式错误或模棱两可怎么办？
    """
    # TODO complete for question 3b
    ANS_RE = re.compile(r"(<[A-Z_]+>)")
    match = ANS_RE.search(judge_output)
    if match:
        match_str: str = match.group(1).strip()
        return match_str
    else:
        return NO_PREFERENCE_FOUND_TAG

def run_llm_judge_eval():
    """
    运行将 LLM 作为评判者的评估，比较模型 E 和 F 在 
    AlpacaEval 数据上的表现：
    使用模型 Z 作为评判者。
    
    对于每条 AlpacaEval 指令，您需要来自模型 E 和 F 的回复，
    然后让评判者进行比较。
    
    请务必保存您的结果（包括模型回复和评判输出）—— 
    您将在后续的 C 和 D 部分中需要这些数据。
    """
    # TODO complete for question 3b
    dataset=load_alpaca_data()
    scores={'E':0,'F':0}
    num=30
    log=[]
    for i in tqdm(range(num)):
        try:
            example=dataset[i]

            instruction=example['instruction']+USE_CHINESE
            query=Query(turns=[
                {"user":instruction}
                ])
            
            response_E=query_model('E',query)
            print_response('E',response_E)
            time.sleep(7)
            response_F=query_model('F',query)
            print_response('F',response_F)
            time.sleep(7)

            prompt=llm_judge_template(query,response_E,response_F)
            query_judge=Query(turns=[
                {"user":prompt},
            ])
            response_Z=query_model('Z',query_judge)
            print_response('Z',response_Z)
            time.sleep(7)

            preference=extract_llm_judge_preference(response_Z.text)
            print(preference)
            if preference == MODEL_E_PREFERED_TAG:
                scores['E']+=1
            elif preference == MODEL_F_PREFERED_TAG:
                scores['F']+=1

            log.append(
                {'E':response_E.text,'F':response_F.text,'Z':response_Z.text,'result':preference}
            )
        except:
            print(f'end at {i}')
            with open('log.jsonl','w',encoding='utf-8') as f:
                for e in log:
                    f.write(json.dumps(e,ensure_ascii=False)+'\n')

    with open('log.jsonl','w',encoding='utf-8') as f:
        for e in log:
            f.write(json.dumps(e,ensure_ascii=False)+'\n')

    print(f"result:\nE:{scores['E']}\nF:{scores['F']}")

    
def plot_model_output_lengths() -> None:
    """
    For Part D: Plot histograms of response lengths for preferred vs. not-preferred outputs.
    """
    # TODO complete for question 3d

    pass

if __name__=="__main__":

    load_dotenv()

    ## Uncomment to run your code
    run_llm_judge_eval()
    #plot_model_output_lengths()
