import random
import numpy as np
import torch
import json
import tqdm
import copy
from datasets import Dataset, interleave_datasets, load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
import os
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def Sky():
    dataset=load_from_disk('../data/public/Sky')
    test_id = 0
    template = ['''作为一个评价专家，给定一个问题和它的两个可能的回答，请选出哪一个回答在连贯性、准确性、覆盖度和上述定义的整体质量方面最为符合。请用JSON格式输出你的判断, 其中"原因"是你提供的解释，"更好的回答"是整数类型的1或2，例如{"原因": "你的解释", "更好的回答": 1}。以下是问题和候选回答的内容：
    \n问题：''',
    '\n回答1：',
    '\n回答2：',]
    tokenizer = AutoTokenizer.from_pretrained("../../open_models/Qwen2-7B-Instruct/")
    i = 0
    all_data = {}
    length_cut = []
    
    for k in ['train']:
        new_data = {'prompt':[], 'gen':[], 'tag':[], 'test_id':[], 'chosen':[], 'q':[], 'r1':[], 'r2':[]}
        for item in dataset[k]:
            if random.random() < 0.5:
                chosen, rejected = 'chosen', 'rejected'
            else:
                chosen, rejected = 'rejected', 'chosen'
            item_prompt = item['chosen'][0]['content']
            prompt = template[0]+item_prompt+template[1]+item[chosen][-1]['content']+template[2]+item[rejected][-1]['content']
            new_data['q'].append(item_prompt)
            new_data['prompt'].append(prompt)
            new_data['gen'].append('')
            new_data['tag'].append('reward-bench')
            new_data['test_id'].append(i)
            new_data['chosen'].append(1 if chosen == 'chosen' else 2)
            new_data['r1'].append(item[chosen][-1]['content'])
            new_data['r2'].append(item[rejected][-1]['content'])
            i += 1
        all_data[k] = Dataset.from_dict(new_data)
    print('length_cut:', np.mean(length_cut))
    all_data = DatasetDict(all_data)
    all_data.save_to_disk('../data/pairwise_critic_inference2_get_answer/Sky')

if __name__ == '__main__':
    Sky()
