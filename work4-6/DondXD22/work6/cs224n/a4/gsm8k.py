import re
import json
from typing import Dict
import os
import time

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model

INVALID_ANS = "[invalid]"

def standard_prompt_template(question: str) -> str:
    """
    Converts a gsm8k question into a standard model input

    Args:
        question: gsm8k question.
    Returns:
        prompt for a model to answer input question.
    """

    prompt = f"""Output a numerical answer to the following problem with two or fewer steps of reasoning. Output your numerical
answer as the only line of your output in the format "#### <numerical_answer>."

Problem: {question}
""".strip()

    return prompt

def standard_output_extractor(model_generation: str) -> str:
    """
    Extracts the string answer from a model generation, assuming it was prompted 
    using a prompt from `standard_prompt_template`.

    Args:
        model_generation: the string generation from the model
    Returns:
        String representing the numerical output of the model for the question, or "[invalid]" if
            no output can be extracted.
    """

    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    match = ANS_RE.search(model_generation)

    if match:
        match_str: str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


# ------------------------------------------- #
# TODO For you to fill in 
# ------------------------------------------- #

def print_response(model_id:str,response:QueryResponse):
    print("=" * 100)
    print(f"Model {model_id} Response")
    print("=" * 100)
    print(f"\n{response.text}\n")
    print("=" * 100)
    print(f"Cost: ${response.cost:.8f}")
    print(f"Tokens used: {response.input_tokens} input, {response.output_tokens} output")
    print("=" * 100)

def eval_model_on_gsm8k() -> None:
    """    
        使用标准提示模板在 GSM8K 数据集上对模型 A 和 B 进行基准测试。

        关于如何查询模型并处理响应，请参阅 example_usage.py。
        数据文件（gsm8k_first_100.jsonl）包含 'question' 和 'numerical_answer' 字段。

        思考：你将使用什么指标来评估性能？你将如何处理模型输出无法解析的情况？
    """
    # TODO complete for question 2bi
    questions=[]
    with open('data\gsm8k_first_100.jsonl','r',encoding='utf-8') as f:
        for line in f:
            item=json.loads(line.strip())
            questions.append(item)
    
    question_num=20
    scores={'A':0,'B':0}
    wrong_answer=[]
    for i in range(question_num):
        question=questions[i]
        prompt=standard_prompt_template(question['question'])
        query=Query(turns=[
            {"user":prompt},
            ])
        
        answer=str(question['numerical_answer'])
        print(f"question:{question['question']}\nanswer:{answer}")

        def eval_one_model(model_id:str):

            response=query_model(model_id,query)
            print_response(model_id,response)

            answer_model=standard_output_extractor(response.text)

            if answer_model == answer:
                scores[model_id]+=1
            else:
                wrong_answer.append({"model_id":model_id,"response":response})
                print("\nWRONG\n")

            time.sleep(10)
        
        eval_one_model('A')
        
        eval_one_model('B')


    print(f"final result:\nscore_A:{scores['A']}\nscoreB:{scores['B']}")

def superior_prompt_template(question: str) -> str:
    """
    Design your own prompt template that outperforms standard_prompt_template on model A.
    
    Args:
        question: gsm8k question.
    Returns:
        Your improved prompt for the model.
    
    Look at standard_prompt_template() to understand the baseline approach. What 
    aspects of how you prompt the model might affect its reasoning or accuracy?
    
    NOTE: Your prompt must still produce output in the "#### <answer>" format
    so that standard_output_extractor() can parse the response.
    """
    # TODO complete for question 2bii
    prompt = f"""Output a numerical answer to the following problem with reasoning as long as you can. Output your numerical
answer as the only line of your output in the format "#### <numerical_answer>."

Problem: {question}
""".strip()

    return prompt
    pass

def eval_model_on_gsm8k_with_improved_prompt() -> None:
    """
    Evaluate model A using your superior_prompt_template.
    """
    # TODO complete for question 2bii
    questions=[]
    with open('data\gsm8k_first_100.jsonl','r',encoding='utf-8') as f:
        for line in f:
            item=json.loads(line.strip())
            questions.append(item)
    
    question_num=20
    scores={'A':0}
    wrong_answer=[]
    for i in range(question_num):
        question=questions[i]
        prompt=superior_prompt_template(question['question'])
        query=Query(turns=[
            {"user":prompt},
            ])
        
        answer=str(question['numerical_answer'])
        print(f"question:{question['question']}\nanswer:{answer}")

        def eval_one_model(model_id:str,query:str):

            response=query_model(model_id,query)
            print_response(model_id,response)

            answer_model=standard_output_extractor(response.text)

            if answer_model == answer:
                scores[model_id]+=1
            else:
                wrong_answer.append({"model_id":model_id,"response":response})
                print("\nWRONG\n")

            time.sleep(10)
        
        eval_one_model('A',query)
    print(f"result:A:{scores['A']}")

if __name__=="__main__":

    load_dotenv()

    # Uncomment to run your code
    eval_model_on_gsm8k()
    eval_model_on_gsm8k_with_improved_prompt()