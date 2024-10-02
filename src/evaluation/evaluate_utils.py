import json
import numpy as np
import re
import demjson
import pandas as pd
import jsonlines

def json_loads(str):
    try:
        return json.loads(str)
    except:
        try:
            return demjson.decode(str)
        except:
            return None

def meta_evaluate(data,):
    def case_our_mode(gen, soft=True):
        json_content = None
        for k in ['更好的回答','更好回答','更好得回答','更好地回答','better_answer','better answer','更好答案','更好得答案','更好的答案','更好地答案','更佳回答','更佳答案',"更好答","最佳答案","更好答 案","更好 的 回答 ","betterAnswer","更好 的 回应 ","更好得回应回答","答案",'回答']:
            if f"{k}" in gen:
                match = re.search(r'"'+k+r'":(.*?)([12１２])', gen, re.DOTALL)
                if match is not None:
                    json_content = {"更好的回答": int(match.group(2))} 
                elif soft:
                    match = re.search(k+r'(.*?)([12１２])', gen, re.DOTALL)
                    if match is not None:
                        json_content = {"更好的回答": int(match.group(2))}
                    else:
                        match = re.search(f'([12])(?=(?:[^1|^2])*?{k})', gen, re.DOTALL)
                        if match is not None:
                            json_content = {"更好的回答": int(match.group(1))}
            if type(json_content) == dict:
                break
        return json_content

    gen = data['gen']
    gen = gen.replace('\n', ' ')
    gen = gen.strip()

    json_content = None
    if "```json" in gen: # 期待的匹配
        match = re.search(r'```json(.*?)```', gen, re.DOTALL)
        if match is not None:
            json_content = json_loads(match.group(1))
            if type(json_content) != dict or '更好的回答' not in json_content.keys():
                match = re.search(r'```json(.*?)```', re.findall(r'```json(.*?)```', gen, re.DOTALL)[-1], re.DOTALL)
                json_content = json_loads(match.group(0)) if match is not None else None
    if type(json_content) != dict and gen.startswith(r'{'): # 直接是一个json
        json_content = json_loads(gen)
    if type(json_content) != dict and '{' in gen and '}' in gen: # 在中间出现
        match = re.search(r'{(.*?)}', gen, re.DOTALL)
        if match is not None:
            json_content = json_loads(match.group(0))
            if json_content is not None and '更好的回答' not in json_content.keys():
                match = re.search(r'{(.*?)}', '{' + re.findall(r'{(.*?)}', gen, re.DOTALL)[-1] + '}', re.DOTALL)
                json_content = json_loads(match.group(0)) if match is not None else None
    if type(json_content) != dict or '更好的回答' not in json_content.keys():
        json_content = case_our_mode(gen)   
    
    if type(json_content) != dict or '更好的回答' not in json_content.keys():
        acc = 0.5
        json_content = {'更好的回答': 'None'}
        # print('--------------')
        # print(gen)
    if json_content['更好的回答'] not in [1,2]:
        acc = 0.5
        json_content = {'更好的回答': 'None'}

    elif data['chosen'] == json_content['更好的回答']:
        acc = 1
    else:
        acc = 0      

    return acc

def evaluate_all(lines, test_id2chosen=None):
    bug_rate = []
    acc = []
    for line in lines:
        if test_id2chosen is not None:
            line['chosen'] = test_id2chosen[line['test_id']]
        result = meta_evaluate(line)
        acc.append(result)
        bug_rate.append(0 if result != 0.5 else 1)
    print('bug rate:', np.mean(bug_rate))
    return acc

def evaluate_all_best_of_n(lines, test_id2chosen=None):
    bug_rate = []
    test_id2acc = {}
    for line in lines:
        result = meta_evaluate(line)
        test_id = line['test_id']
        if test_id not in test_id2acc.keys():
            test_id2acc[test_id] = []
        test_id2acc[test_id].append(result)
        bug_rate.append(0 if result != 0.5 else 1)
    print('bug rate:', np.mean(bug_rate))
    def major_vote(item):
        if np.mean(item) > 0.5:
            return 1
        elif np.mean(item) < 0.5:
            return 0
        return 0.5
    acc = [major_vote(item) for item in list(test_id2acc.values())]
    return acc

def load_file(base_path):
    return [json.loads(line) for line in open(base_path).readlines()]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='', type=str)
    parser.add_argument('--ref_path', default='', type=str)
    args = parser.parse_args()
    
    if args.ref_path != '':
        test_id2chosen = {}
        for item in load_file(args.ref_path):
            test_id2chosen[item['test_id']] = item['chosen']
    else:
        test_id2chosen = None

    if 'best' not in args.path:
        acc = evaluate_all(load_file(args.path), test_id2chosen=test_id2chosen)
    else:
        acc = evaluate_all_best_of_n(load_file(args.path), test_id2chosen=test_id2chosen)

    print('mean acc:', np.mean(acc))

