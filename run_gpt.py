import openai
import os,re
from time import sleep,time
import itertools
from collections import Counter,defaultdict
import numpy as np
import json
from tqdm import tqdm

# 设置代理
os.environ['http_proxy'] = <local_proxy_link>
os.environ['https_proxy'] = <local_proxy_link>

# 添加api key
openai.api_key = <your_api_keys>

def chatgpt(content, temperature,dataset_name,shot_filepath=False):
    delimiter = '```'
    my_prompt1 = f'''
    You are an excellent linguist, and you can recognize the named entities from\
    given Twitter texts. 
    Output a JSON object contains the following keys: \
    'Location', 'Group', 'Person', 'Creative work', 'Corporation' and 'Product'.
     The value should be in the format of 'list'.

    Here is some explanation for the keys: 
    --'Location' means some place, e.g. 'Sonmarg' is a named entity of Location.
    --'Group' means the name of groups like music band, sports team, e.g. 'Avalanche Rescue Teams' is a named entity of Group.
    --'Person' means the name of someone, e.g. 'Colonel Rajesh Kalia' is a named entity of Person.
    --'Creative work' means creative works, e.g. 'What Else is Making News' is a named entity of Creative work.
    --'Corporation' means corporations, e.g. 'CVS' is a named entity of Corporation.
    --'Product' means products, e.g. 'epipen' is a named entity of Product.
    These examples is just to help you understand the key.

    Your answer MUST be a JSON object even if no named entity is found.
    '''

    my_prompt2 = f'''
    You are an excellent linguist, and you can recognize the named entities from\
    given Twitter texts. 
    Output a JSON object contains the following keys: \
    'PER', 'LOC', 'ORG' and 'OTHER'. The value should be in the format of 'list'.

    Here is some explanation for the keys: 
    --'PER' means person, e.g. 'CLINTONS' is a named entity of PER.
    --'LOC' means location, e.g. 'Fayetteville' is a named entity of LOC.
    --'ORG' means organization, e.g. 'Yamaha Motor' is a named entity of ORG.
    --'OTHER' means other types like music, product, brand and so on. e.g. 'ClintonCash' is a named entity of OTHER.
    These examples is just to help you understand what the keys mean.

    You should provide named entity with correct length of words, and try to find \
        named entity from the text delimited by {delimiter}.\
    You should strictly follow the format of JSON even if no named entity is recognized from the given sentence.
    '''
    my_prompt3 = f'''
    You are an excellent linguist and biologist, and you can recognize the named entities from\
    given texts which focus on technical terms in the biology domain.
    Output a JSON object contains the following keys: 'DNA', 'protein', 'cell-type', 'cell-line' and 'RNA' \
    and the value should be a list of entity, like '[a,b,...]'

    Here is some explanation for the keys: 
    --'DNA' means DNA, e.g. 'kappa B core sequence' is a named entity of DNA.
    --'protein' means protein, e.g. 'glucocorticoid receptors' is a named entity of protein.
    --'cell_type' means cell type, e.g. 'T cell' is a named entity of cell_type.
    --'cell_line' means cell line, e.g. 'Hella cells' is a named entity of cell_line.
    --'RNA' means RNA, e.g. 'ER mRNA' is a named entity of RNA.
    These examples is just to help you understand what the keys mean.

    You should provide named entity with correct length of words, and try to find all \
    of named entity in the sentence delimited by {delimiter}.\

    '''

    my_prompt = {'WNUT2017':my_prompt1,'Twitter':my_prompt2,'Bio-NER':my_prompt3}

    if not shot_filepath:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        	{"role": "system", "content": my_prompt[dataset_name]},
            {"role": "user", "content": f'You should provide named entity with correct length of words, and try to find \
             named entity in the sentence delimited by {delimiter}.\n{delimiter}\n{content}\n{delimiter}'},
            # {"role": "user", "content": f'{delimiter}\n{content}\n{delimiter}'},
        ],
        temperature = temperature,
        top_p=0.5,
        frequency_penalty=0,
        )

    else:
        with open(shot_filepath,'r',encoding='utf-8') as shot_f:
            prompt_data = json.load(shot_f)

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
        	{"role": "system", "content": my_prompt[dataset_name]},
            {"role": "user", "content": prompt_data[0]['TEXT']},
            {"role": "assistant", "content": str(prompt_data[0]['NEs'])},
            {"role": "user", "content": prompt_data[1]['TEXT']},
            {"role": "assistant", "content": str(prompt_data[1]['NEs'])},
            {"role": "user", "content": prompt_data[2]['TEXT']},
            {"role": "assistant", "content": str(prompt_data[2]['NEs'])},
            {"role": "user", "content": prompt_data[3]['TEXT']},
            {"role": "assistant", "content": str(prompt_data[1]['NEs'])},
            {"role": "user", "content": prompt_data[4]['TEXT']},
            {"role": "assistant", "content": str(prompt_data[2]['NEs'])},
            {"role": "user", "content": f'You should provide named entity with correct length of words, and try to find \
             named entity in the sentence delimited by {delimiter}.\n{delimiter}\n{content}\n{delimiter}'},
            # {"role": "user", "content": f'Are you sure? Check your answer and output your new answer in JSON again.'},
        ],
        temperature = 0,
        top_p=0.5,
        frequency_penalty=0,
        )
    return response
    # return response.choices[0].message.content

if __name__=='__main__':

    # dataset_name = 'WNUT2017'
    dataset_name = 'Twitter'
    # dataset_name = 'Bio-NER'
    # file_path = f'./data/{dataset_name}/test.txt'
    output_directory = f'./output/{dataset_name}'
    temp_filepath = output_directory+'/query.json'
    shot_filepath = output_directory+'/5shot.json'
    output_filepath = output_directory+'/result1.json'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

## 保存到.json中
    # test_words,test_labels = load_file(file_path)
    # reshape_data_new(test_words,test_labels,temp_filepath)

## 读取json
    with open(temp_filepath,'r',encoding='utf-8') as f:
        data = json.load(f)


## 保存
    token_num = 0
    texts = []
    results = []
    pbar = tqdm(range(len(data)),desc=f'Asking gpt api, please wait')
    for i in pbar:
        pre_time = time()
        try:
            response = chatgpt(content=data[i]['TEXT'],temperature=0,dataset_name=dataset_name)
            # response = chatgpt(content=data[i]['TEXT'],temperature=0,dataset_name=dataset_name,shot_filepath=shot_filepath)
        except Exception as er:
            new_time = time()           
            print(er)
        finally:
            new_time = time()
            time_diff = new_time-pre_time
            token_num += response.usage.total_tokens
            result = response.choices[0].message.content
            results.append(result)
            # results.append(json.loads(response.choices[0].message.content))
            pbar.desc = f'current token usage is {token_num}'
            sleep(max(21-time_diff,1)) ##确保请求频率不过高

    with open(output_filepath,'w',encoding='utf-8') as f:
        json.dump(results,f,indent=4,ensure_ascii=False)
