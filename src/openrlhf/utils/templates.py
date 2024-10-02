from datasets import Dataset
import re

templates = {
    'guide': ['''作为一个评价专家，给定一个问题和它的两个可能的回答，请分析二者在连贯性、准确性、覆盖度和上述定义的整体质量方面的表现。
        \n问题：''',
        '\n回答1：',
        '\n回答2：',
        '''
        已知回答''',
        "更好，请说明原因:"
    ],
    'guide_reverse': ['''作为一个评价专家，给定一个问题和它的两个可能的回答，请分析二者在连贯性、准确性、覆盖度和上述定义的整体质量方面的表现。
        \n问题：''',
        '\n回答1：',
        '\n回答2：',
        '''
        已知回答''',
        "更好，请说明原因:"
    ],
    'remove_format':[
        '''作为一个评价专家，给定一个问题和它的两个可能的回答，请选出哪一个回答在准确性上最佳，并且评价的时候不要考虑内容的表达方式。请用JSON格式输出你的判断, 其中"原因"是你提供的解释，"更好的回答"是整数类型的1或2，例如{"原因": "你的解释", "更好的回答": 1}。以下是问题和候选回答的内容：
        \n问题：''',
        '\n回答1：',
        '\n回答2：',
    ],
    'enhance_format':[
        '''作为一个评价专家，给定一个问题和它的两个可能的回答，请选出哪一个回答最佳，评价的时候需要多考虑内容的表达方式，一般认为表达更通俗的内容更好。请用JSON格式输出你的判断, 其中"原因"是你提供的解释，"更好的回答"是整数类型的1或2，例如{"原因": "你的解释", "更好的回答": 1}。以下是问题和候选回答的内容：
        \n问题：''',
        '\n回答1：',
        '\n回答2：',
    ],
    'remove_format_guide':[
        '''作为一个评价专家，给定一个问题和它的两个可能的回答，请选出哪一个回答在准确性上最佳，并且评价的时候不要考虑内容的表达方式。以下是问题和候选回答的内容：
        \n问题：''',
        '\n回答1：',
        '\n回答2：',
        '''
        已知回答''',
        "更好，请说明原因:"
    ],
    'remove_format_guide_reverse':[
        '''作为一个评价专家，给定一个问题和它的两个可能的回答，请选出哪一个回答在准确性上最佳，并且评价的时候不要考虑内容的表达方式。以下是问题和候选回答的内容：
        \n问题：''',
        '\n回答1：',
        '\n回答2：',
        '''
        已知回答''',
        "更好，请说明原因:"
    ],
}

modification_templates = {
    'guide': [
        '''{"原因":"''',
        '''","更好的回答": ''',
        '''}'''
    ],
    'guide_reverse': [
        '''{"原因":"''',
        '''","更好的回答": ''',
        '''}'''
    ],
    'remove_format_guide':[
        '''{"原因":"''',
        '''","更好的回答": ''',
        '''}'''
    ],
    'remove_format_guide_reverse': [
        '''{"原因":"''',
        '''","更好的回答": ''',
        '''}'''
    ],
}

def change_data_templates(dataset, template_key, reverse=False):
    template = templates[template_key]
    new_dataset = {}
    for k in dataset[0].keys():
        new_dataset[k] = []
    new_dataset['gen'] = []
    for item in dataset:
        item['chosen'] = item['chosen_id'] if 'chosen_id' in item.keys() else item['chosen']
        if 'q' not in item.keys():
            pattern = r"问题：(.*?)回答1：(.*?)回答2：(.*)"
            match = re.search(pattern, item['prompt'], re.DOTALL)
            item['r1'] = match.group(2).strip()
            item['r2'] = match.group(3).strip()
            item['q'] = match.group(1).strip()
        prompt = template[0]+item['q']+template[1]+item['r1']+template[2]+item['r2']
        if 'guide' in template_key:
            if reverse == False:
                gen = template[3] + str(item['chosen']) + template[4]
            else:
                gen = template[3] + str(3-item['chosen']) + template[4]
        else:
            gen = ''
        new_dataset['prompt'].append(prompt)
        new_dataset['gen'].append(gen)
        for k in new_dataset.keys():
            if k not in ['prompt','gen']:
                new_dataset[k].append(item[k])
    return Dataset.from_dict(new_dataset)

def reshape_output(output, template_key):
    if 'guide_reverse' in template_key:
        template = modification_templates[template_key]
        return template[0] + output['原因'] + template[1] + str(3-output['chosen']) + template[2]
    elif 'guide' in template_key:
        template = modification_templates[template_key]
        return template[0] + output['原因'] + template[1] + str(output['chosen']) + template[2]
    else:
        return output['原因']

