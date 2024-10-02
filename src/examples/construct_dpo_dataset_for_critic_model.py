import os
from datasets import Dataset, DatasetDict
import argparse
import json
import random
import re
import numpy as np
import copy
from datasets import load_from_disk
random.seed(2021)

def evaluate(input_str, save_acc=False):
    def json_loads(str):
        try:
            return json.loads(str)
        except:
            try:
                return demjson.decode(str)
            except:
                return None

    def case_our_mode(gen):
        json_content = None
        if "更好的回答" in gen:
            re.search(r'"更好的回答":(.*?)([12])', gen, re.DOTALL)
            match = re.search(r'"更好的回答":(.*?)([12])', gen, re.DOTALL)
            if match is not None:
                json_content = {"更好的回答": int(match.group(2))}
        return json_content

    json_content = None
    input_str = input_str.strip()
    input_str = input_str.replace('”，','",').replace('”,','",').replace('”,','",').replace('”，','",')
    if "```json" in input_str: # 期待的匹配
        match = re.search(r'```json(.*?)```', input_str, re.DOTALL)
        if match is not None:
            json_content = json_loads(match.group(1))
            if json_content is not None and type(json_content) == dict and '更好的回答' not in json_content.keys():
                match = re.search(r'```json(.*?)```', re.findall(r'```json(.*?)```', input_str, re.DOTALL)[-1], re.DOTALL)
                json_content = json_loads(match.group(0)) if match is not None else None
    if json_content is None and input_str.startswith(r'{'): # 直接是一个json
        json_content = json_loads(input_str)
    if json_content is None and '{' in input_str and '}' in input_str: # 在中间出现
        match = re.search(r'{(.*?)}', input_str, re.DOTALL)
        if match is not None:
            json_content = json_loads(match.group(0))
            if json_content is not None and '更好的回答' not in json_content.keys():
                match = re.search(r'{(.*?)}', '{' + re.findall(r'{(.*?)}', input_str, re.DOTALL)[-1] + '}', re.DOTALL)
                json_content = json_loads(match.group(0)) if match is not None else None
    if json_content is None:
        json_content = case_our_mode(input_str) 
    if type(json_content) == str:
        try:
            json_content = json_loads(json_content)
        except:
            json_content = None
    if json_content is None or type(json_content) != dict or '更好的回答' not in json_content.keys() or json_content['更好的回答'] not in [1,2]:
        json_content = {'更好的回答': 'None'}
    return json_content['更好的回答']

def load_datasets(dataset_list):
    re = []
    for dataset in dataset_list:
        test_id2data = {}
        for item in [json.loads(line) for line in open(dataset).readlines()]:
            if item['test_id'] not in test_id2data.keys():
                test_id2data[item['test_id']] = []
            test_id2data[item['test_id']].append(item)
        re.append(test_id2data)
    return re

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('../../../open_models/Qwen2-7B-Instruct')

def get_content_before_last_period(text):
    # 找到最后一个句号的位置
    last_period_index = max(text.rfind('。'), text.rfind('.'))
    
    # 如果找到了句号，返回句号前的内容
    if last_period_index != -1:
        return text[:last_period_index]
    else:
        # 如果没找到句号，返回原文本
        return -1

def split_positive_negative(test_id2data, length_check=False):
    test_id2data_new = {}
    none_rate = []
    for test_id in test_id2data.keys():
        test_id2data_new[test_id] = {'positive':[], 'negative':[], 'unknown':[]}
        for item in test_id2data[test_id]:
            evaluate_result = evaluate(item['gen'])
            if length_check:
                try:
                    if len(item['gen']) > 600:
                        json_dict = json.loads(item['gen'])
                        encode_info = tokenizer.encode(json_dict['原因'])
                        if len(encode_info) > 400:
                            # if evaluate_result != 'None':
                            #     item2 = copy.deepcopy(item)
                            #     json_dict = json.loads(item2['gen'])
                            #     json_dict
                            #     ['原因']
                            #     test_id2data_new[test_id][evaluate_result].append(item2)
                            # evaluate_result = 'negative'
                            # item['gen'] = tokenizer.decode(encode_info[:512])
                            json_dict['原因'] = get_content_before_last_period(tokenizer.decode(encode_info[:400]))
                            if json_dict['原因'] == -1:
                                evaluate_result = 'negative'
                                item['gen'] = tokenizer.decode(encode_info[:512])
                            else:
                                item['gen'] = json.dumps(json_dict, ensure_ascii=False)
                except:
                    pass
            if evaluate_result == item['chosen']:
                test_id2data_new[test_id]['positive'].append(item)
                none_rate.append(0)
            elif evaluate_result == 'None':
                test_id2data_new[test_id]['unknown'].append(item)
                none_rate.append(1)
            else:
                test_id2data_new[test_id]['negative'].append(item)
                none_rate.append(0 if evaluate_result != 'negative' else 1)
    print('none_rate:', np.mean(none_rate))

    return test_id2data_new

def add_test_id2name(dataset_dict):
    test_id2data_name = {}
    for name in ['train','test','test2']:
        if name in dataset_dict.keys():
            tmp_data = dataset_dict[name]
            for item in tmp_data:
                test_id2data_name[item['test_id']] = name
    return test_id2data_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_of_n', default=8, type=int)
    parser.add_argument('--task', default='数学计算', type=str)
    parser.add_argument('--reference_path', default='', type=str)
    parser.add_argument('--base_path', default='', type=str)
    parser.add_argument('--prefix', default='', type=str)
    args = parser.parse_args()
    reference_path = f'../../outputs/inference/zeroshot_llm_{args.task}/' if args.reference_path == '' else args.reference_path
    path_best_of_n = f'{reference_path}/best_of_{args.best_of_n}.jsonl'
    path_good = f'{reference_path}/guide.jsonl'
    path_bad = f'{reference_path}/guide_reverse.jsonl'
    
    path_best_of_n, path_good, path_bad = load_datasets([path_best_of_n, path_good, path_bad])
    
    path_best_of_n = split_positive_negative(path_best_of_n)
    path_good = split_positive_negative(path_good, length_check=True)
    path_bad = split_positive_negative(path_bad, length_check=True)

    output = {'conversations':[],'chosen':[],'rejected':[],'test_id':[],'count':[],'type':[],'tag':[], 'chosen_id':[],'pair_type':[]}
    base_data_path=f'../../../data/sub_data/pairwise_critic_inference2_get_answer/{args.task}' if args.base_path == '' else args.base_path
    test_id2prompt = {}
    dataset = load_from_disk(base_data_path)
    test_id2info = add_test_id2name(dataset)

    # 使用 map 方法添加新的列
    def extract_prompt(example):
        # 从 conversations 列中提取第一个字典的 'value'
        return {'prompt': example['conversations'][0]['value']}
        
    for k in dataset.keys():
        if 'prompt' not in dataset[k][0].keys():
            dataset[k] = dataset[k].map(extract_prompt)
        for item in dataset[k]:
            test_id2prompt[item['test_id']] = item['prompt']
        
    construct_pair_success = []
    num_samples = {
        'best_of_n':0,
        'best_of_n_positive2unknown':0,
        'preamble':0,
        'preamble2unknown':0,
        'num_lost':0,
        'fake':0
    }
    index2type = []
    best_of_n_type = {'ok':0,'all_positive':0,'all_negative':0}
    for test_id in path_best_of_n.keys():
        random.shuffle(path_best_of_n[test_id]['positive'])
        random.shuffle(path_best_of_n[test_id]['negative'])
        construct_pair_success.append(0)
        if len(path_best_of_n[test_id]['positive']) > 0 and len(path_best_of_n[test_id]['negative']) > 0:
            best_of_n_type['ok'] += 1
        elif len(path_best_of_n[test_id]['positive']) > 0:
            best_of_n_type['all_positive'] += 1
        elif len(path_best_of_n[test_id]['negative']) > 0:
            best_of_n_type['all_negative'] += 1
        if len(path_best_of_n[test_id]['positive']) > 0 and len(path_best_of_n[test_id]['negative']) > 0:
            for idx in range(min(len(path_best_of_n[test_id]['positive']), len(path_best_of_n[test_id]['negative']), 3)):
                item = path_best_of_n[test_id]['positive'][idx]
                item2 = path_best_of_n[test_id]['negative'][idx]
                output['count'].append(1)
                output['type'].append('None')
                output['conversations'].append([{'from':'human', 'value':test_id2prompt[item['test_id']]}])
                output['test_id'].append(test_id)
                output['chosen'].append(item['gen'])
                output['rejected'].append(item2['gen'])
                output['tag'].append(item['tag'])
                output['chosen_id'].append(item['chosen'])
                output['pair_type'].append('best_of_n')
                construct_pair_success[-1] += 1
                num_samples['best_of_n'] += 1
                index2type.append('best_of_n')
        if len(path_best_of_n[test_id]['positive']) > 0 and len(path_best_of_n[test_id]['unknown']) > 0:
            for idx in range(min(len(path_best_of_n[test_id]['positive']), len(path_best_of_n[test_id]['unknown']), 1)):
                item = path_best_of_n[test_id]['positive'][idx]
                item2 = path_best_of_n[test_id]['unknown'][idx]
                output['count'].append(1)
                output['type'].append('None')
                output['conversations'].append([{'from':'human', 'value':test_id2prompt[item['test_id']]}])
                output['test_id'].append(test_id)
                output['chosen'].append(item['gen'])
                output['rejected'].append(item2['gen'])
                output['chosen_id'].append(item['chosen'])
                output['tag'].append(item['tag'])
                construct_pair_success[-1] += 1
                output['pair_type'].append('best_of_n_positive2unknown')
                num_samples['best_of_n_positive2unknown'] += 1
                index2type.append('best_of_n_positive2unknown')
        if len(path_good[test_id]['positive']) > 0 and len(path_bad[test_id]['negative']) > 0:
            item = path_good[test_id]['positive'][0]
            item2 = path_bad[test_id]['negative'][0]
            output['count'].append(1)
            output['type'].append('None')
            output['conversations'].append([{'from':'human', 'value':test_id2prompt[item['test_id']]}])
            output['test_id'].append(test_id)
            output['chosen'].append(item['gen'])
            output['rejected'].append(item2['gen'])
            output['tag'].append(item['tag'])
            output['chosen_id'].append(item['chosen'])
            construct_pair_success[-1] += 1
            output['pair_type'].append('preamble')
            num_samples['preamble'] += 1
            index2type.append('preamble')
        elif len(path_good[test_id]['negative']) == 0 and len(path_bad[test_id]['positive']) == 0:
            item = path_good[test_id]['positive'][0] if len(path_good[test_id]['positive']) > 0 else path_good[test_id]['unknown'][0]
            item2 = path_bad[test_id]['negative'][0] if len(path_bad[test_id]['negative']) > 0 else path_bad[test_id]['unknown'][0]
            output['count'].append(1)
            output['type'].append('None')
            output['conversations'].append([{'from':'human', 'value':test_id2prompt[item['test_id']]}])
            output['test_id'].append(test_id)
            output['chosen'].append(item['gen'])
            output['rejected'].append(item2['gen'])
            output['tag'].append(item['tag'])
            output['chosen_id'].append(item['chosen'])
            construct_pair_success[-1] += 1
            num_samples['preamble2unknown'] += 1
            output['pair_type'].append('preamble2unknown')
            index2type.append('preamble2unknown')
        if construct_pair_success[-1] == 0:
            if construct_pair_success[-1] == 0: 
                num_samples['num_lost'] += 1
    construct_pair_success = np.array(construct_pair_success)
    print('successful prompts:', np.sum(construct_pair_success>0)/len(construct_pair_success))
    print('mean paris for each prompt:', np.sum(construct_pair_success)/len(construct_pair_success))       
    print('length of test_ids:', len(path_best_of_n.keys()))

    dataset_dict = {'train':dict({k:[] for k,v in output.items()}),'validation':dict({k:[] for k,v in output.items()}),'test':dict({k:[] for k,v in output.items()}),'test2':dict({k:[] for k,v in output.items()})}
    for i in range(len(output['count'])):
        for k in output.keys():
            dataset_dict[test_id2info[output['test_id'][i]]][k].append(output[k][i])

    dataset_dict['validation'] = dataset_dict['test']
    
    dataset_dict = DatasetDict({k:Dataset.from_dict(v) for k,v in dataset_dict.items()}) 

    if 'dpo' not in reference_path:
        reference_path += '/dpo'
    dataset_dict.save_to_disk(f'{reference_path}/dataset/')
    