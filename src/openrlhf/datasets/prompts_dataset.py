from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none

def preprocess_data(data, input_template=None, input_key=None) -> str:
    # custom dataset
    if input_key:
        prompt = data[input_key]
    else:
        # Dahoas/full-hh-rlhf
        if exist_and_not_none(data, "prompt"):
            prompt = data["prompt"]
            # tasksource/oasst1_pairwise_rlhf_reward
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "
                )
            input_template = None  # do not modified with input template again
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + "\n" + data["question"]
        # BelleGroup/train_0.5M_CN
        # LLMs/Alpaca-ShareGPT
        # yahma/alpaca-cleaned
        # QingyiSi/Alpaca-CoT
        elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
            input = " " + data["input"] if exist_and_not_none(data, "input") else ""
            prompt = data["instruction"] + input
        # lmsys/chatbot_arena_conversations
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))
                    else:
                        result.append(l["content"])
                return "\n".join(result)

            prompt = data["conversation_a"][:-1]
            prompt = process_chatbot_arena_conversations(prompt)
            input_template = None  # do not modified with input template again
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "question") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
        else:
            raise ValueError("Unknown prompts dataset")

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt

# def process_prompt_data(data, user_token = "<reserved_106>", assistant_token = "<reserved_107>"):
#     prompt = ""
#     for i, message in enumerate(data['conversations']):
#         if i % 2 == 0:
#             assert message["from"] == "human"
#             prompt += user_token + message["value"]
#         else:
#             assert message["from"] == "gpt"
#             prompt += assistant_token + message["value"]
#     if data['conversations'][-1]["from"] == "human":
#         prompt = prompt + assistant_token
#     return prompt

def process_prompt_data(data, tokenizer, input_template, add_generation_prompt=False):
    try:
        prompt = tokenizer.apply_chat_template(data, add_generation_prompt=add_generation_prompt, tokenize=False)
    except:
        for item in data:
            if item['role'] == 'human':
                item['role'] = 'user'
        prompt = tokenizer.apply_chat_template(data, add_generation_prompt=add_generation_prompt, tokenize=False)
    if data[-1]['role'] in ['assistant','gpt'] and 'Human' in input_template and ('Assistant' in input_template or 'reason' in input_template or 'judgment' in input_template):
        prompt = prompt[:-11]
    elif 'Human' in input_template:
        prompt += 'Assistant:'
    return prompt

def process_data(data, tokenizer, input_template):
    # 处理 meta 信息
    tmp_meta = {k: data[k] for k in ['tag', 'test_id', 'chosen', 'chosen_id'] if k in data}
    if 'chosen_id' in tmp_meta.keys():
        tmp_meta['chosen'] = tmp_meta['chosen_id']
    # 处理 prompt 数据
    if 'Human' in input_template and 'Assistant' in input_template:
        data = [{'role': 'human', 'content': data['prompt']}, {'role': 'assistant', 'content': data['gen']}]
    elif 'Human' in input_template and 'reason' in input_template:
        data = [{'role': 'human', 'content': data['prompt']}, {'role': 'assistant', 'content': ' {\"原因\":'}]
    elif 'Human' in input_template and 'judgment' in input_template:
        data = [{'role': 'human', 'content': data['prompt']}, {'role': 'assistant', 'content': ' {\"更好的回答\":'}]
    else:
        data = [{'role': 'human', 'content': data['prompt']}]

    prompt = process_prompt_data(data, tokenizer, input_template)
    return tmp_meta, prompt

def process_data_raw(data, tokenizer, input_template):
    # 处理 meta 信息
    tmp_meta = {k: data[k] for k in ['tag', 'test_id', 'chosen', 'chosen_id'] if k in data}
    if 'chosen_id' in tmp_meta.keys():
        tmp_meta['chosen'] = tmp_meta['chosen_id']
    return tmp_meta, process_prompt_data(data['prompt'], tokenizer, input_template, add_generation_prompt=True)

class MyPromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
        multi_thread = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        #self.input_template = input_template
        #input_key = getattr(self.strategy.args, "input_key", None)
        self.dataset = dataset
        # self.huancun = True if len(dataset) > 20000 else False
        self.huancun = False
        # if self.huancun:
        #     return

        if multi_thread:
            import concurrent.futures
            import multiprocessing
            # 多进程处理
            manager = multiprocessing.Manager()
            prompts = manager.list()
            meta_info = manager.list()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_data, data, tokenizer, input_template) for data in tqdm(dataset, disable=not strategy.is_rank_0())]
                
                for future in concurrent.futures.as_completed(futures):
                    tmp_meta, prompt = future.result()
                    meta_info.append(tmp_meta)
                    prompts.append(prompt)

            self.prompts = list(prompts)
            self.meta_info = list(meta_info)
        else:
            self.prompts = []
            self.meta_info = []
            for data in dataset:
                if input_template == 'raw':
                    tmp_meta, prompt = process_data_raw(data, tokenizer, input_template)
                else:
                    tmp_meta, prompt = process_data(data, tokenizer, input_template)
                self.meta_info.append(tmp_meta)
                self.prompts.append(prompt)

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        if self.huancun:
            return process_data(self.dataset[idx], self.tokenizer, self.input_template)
        else:
            return self.prompts[idx], self.meta_info[idx]

class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        #self.input_template = input_template
        #input_key = getattr(self.strategy.args, "input_key", None)

        self.prompts = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            #prompt = preprocess_data(data, input_template, input_key)
            prompt = process_prompt_data(data)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
