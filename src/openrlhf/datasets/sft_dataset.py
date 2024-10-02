from typing import Callable
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences


def preprocess_data(data, input_template=None, input_key=None, output_key=None):
    # custom dataset
    if input_key and output_key:
        prompt = data[input_key]
        target = data[output_key]
    else:
        # Dahoas/full-hh-rlhf
        # iamketan25/open-assistant-instructions
        if exist_and_not_none(data, "prompt") and exist_and_not_none(data, "chosen"):
            prompt = data["prompt"]
            target = data["chosen"]
            input_template = None  # do not modified with input template again
        # pvduy/sharegpt_alpaca_oa_vicuna_format
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "label"):
            prompt = data["prompt"].replace("USER:", "").replace("ASSISTANT:", "")
            target = data["label"].replace("</s>", "")
        # BelleGroup/train_0.5M_CN
        # LLMs/Alpaca-ShareGPT
        # yahma/alpaca-cleaned
        # QingyiSi/Alpaca-CoT
        elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
            input = " " + data["input"] if exist_and_not_none(data, "input") else ""
            prompt = data["instruction"] + input
            target = data["output"]
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            prompt = data["system_prompt"] + "\n" + data["question"]
            target = data["response"]
        # crumb/gpt4all-clean
        # nomic-ai/gpt4all-j-prompt-generations
        elif exist_and_not_none(data, "prompt") and exist_and_not_none(data, "response"):
            prompt = data["prompt"]
            target = data["response"]
        # EleutherAI/pile [pretrain !!!]
        elif exist_and_not_none(data, "text") and exist_and_not_none(data, "meta"):
            assert input_template is None  # pretrain_mode
            prompt = ""
            target = data["text"]
        # for batch_inference.py
        elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
            prompt = data["input"]
            target = data["output"]
            input_template = None
        else:
            raise ValueError("Unknown SFT dataset")

    # input template
    if input_template:
        prompt = input_template.format(prompt)
    return prompt, target

def process_sft_data_gen2(data, tokenizer, input_template, template_length = 27):
    prompt = tokenizer.apply_chat_template(data, add_generation_prompt=False, tokenize=False)
    target = data[-1]
    target = prompt[-template_length:]
    prompt = prompt[:-template_length]
    return prompt, target

def process_sft_data(data, tokenizer, input_template):
    prompt = tokenizer.apply_chat_template(data[:-1], add_generation_prompt=True, tokenize=False)
    target = data[-1]["content"]
    return prompt, target

# def process_sft_data(data, user_token = "<reserved_106>", assistant_token = "<reserved_107>"):
#     prompt = ""
#     assert data['conversations'][-1]["from"] == "gpt"
#     for i, message in enumerate(data['conversations']):
#         if i == len(data['conversations']) - 1:
#             break
#         if i % 2 == 0:
#             assert message["from"] == "human"
#             prompt += user_token + message["value"]
#         else:
#             assert message["from"] == "gpt"
#             prompt += assistant_token + message["value"]

#     prompt = prompt + assistant_token
#     target = data['conversations'][-1]["value"]
#     return prompt, target

class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
        pretrain_mode=False,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.targets = []
        self.prompt_ids_lens = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        #input_key = getattr(self.strategy.args, "input_key", None)
        #output_key = getattr(self.strategy.args, "output_key", None)

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            #prompt, target = preprocess_data(data, None if pretrain_mode else input_template, input_key, output_key)
            prompt, target = process_sft_data(data)
            if not self.pretrain_mode:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            else:
                prompt_ids_len = 0

            if not self.pretrain_mode:
                # filter the sample whose length is greater than max_length (2 for answer length)
                if prompt_ids_len >= self.max_length - 2:
                    continue
                if not prompt or not target:
                    continue

            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.targets.append(target)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        target = self.targets[idx]

        input_token = self.tokenizer(
            #prompt + target + " " + self.tokenizer.eos_token,
            prompt + target + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        info = {"input": prompt, "output": target}
        # to avoid EOS_token truncation
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        for prompt_ids_len, input_id, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos


class MySFTDataset(Dataset):
    def pack_data(self, prompt, target):
        if not self.pretrain_mode:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
        else:
            prompt_ids_len = 0

        if not self.pretrain_mode:
            # filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                return None, None, None
            if not prompt or not target:
                return None, None, None
        return prompt, target, prompt_ids_len

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human: {}\nAssistant: ",
        pretrain_mode=False,
        eval_mode=False,
        gen2 = False,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.targets = []
        self.prompt_ids_lens = []
        self.meta_infos = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.eval_mode = eval_mode
        self.gen2 = gen2
        if self.gen2:
            self.prompts2 = []
            self.targets2 = []
            self.prompt_ids_lens2 = []

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            messages = {}
            messages = [{'role':'human', 'content': data['prompt']},{'role':'assistant', 'content': data['gen']}]
            prompt, target = process_sft_data(messages, tokenizer, input_template)
            prompt, target, prompt_ids_len = self.pack_data(prompt, target)
            if gen2:
                prompt2, target2 = process_sft_data_gen2(messages, tokenizer, input_template)
                prompt2, target2, prompt_ids_len2 = self.pack_data(prompt2, target2)
            if prompt is None or (self.gen2 and prompt2 is None):
                continue

            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.targets.append(target)
            if gen2:
                self.prompt_ids_lens2.append(prompt_ids_len2)
                self.prompts2.append(prompt2)
                self.targets2.append(target2)
            # ziyi add meta info
            meta_info = {}
            for k in ['prompt', 'gen', 'test_id', 'chosen', 'tag',]:
                if k in data.keys():
                    meta_info[k] = data[k]
            self.meta_infos.append(meta_info)

    def __len__(self):
        length = len(self.prompts)
        return length

    def get_item_from_saved(self, prompt_ids_len, prompt, target):
        if self.eval_mode == False:
            input_token = self.tokenizer(
                prompt + target + self.tokenizer.eos_token,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
        else:
            input_token = self.tokenizer(
                prompt + target + self.tokenizer.eos_token,
                return_tensors="pt",
            )
        info = {"input": prompt, "output": target}
        # to avoid EOS_token truncation
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    def __getitem__(self, idx):
        if self.gen2:
            prompt_ids_len, input_ids, attention_mask, info = self.get_item_from_saved(self.prompt_ids_lens[idx], self.prompts[idx], self.targets[idx])
            prompt_ids_len2, input_ids2, attention_mask2, _ = self.get_item_from_saved(self.prompt_ids_lens2[idx], self.prompts2[idx], self.targets2[idx])
            return prompt_ids_len, input_ids, attention_mask, info, prompt_ids_len2, input_ids2, attention_mask2
        else:
            return self.get_item_from_saved(self.prompt_ids_lens[idx], self.prompts[idx], self.targets[idx])

    def collate_fn(self, item_list):
        if self.gen2:
            prompt_ids_lens = []
            prompt_ids_lens2 = []
            input_ids = []
            input_ids2 = []
            attention_masks = []
            attention_masks2 = []
            infos = []

            for prompt_ids_len, input_id, attention_mask, info, prompt_ids_len2, input_id2, attention_mask2 in item_list:
                prompt_ids_lens.append(prompt_ids_len)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                infos.append(info)
                prompt_ids_lens2.append(prompt_ids_len2)
                input_ids2.append(input_id2)
                attention_masks2.append(attention_mask2)

            input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
            input_ids2 = zero_pad_sequences(input_ids2, "right", self.tokenizer.pad_token_id)

            attention_masks = zero_pad_sequences(attention_masks, "right")
            attention_masks2 = zero_pad_sequences(attention_masks2, "right")

            return prompt_ids_lens, input_ids, attention_masks, infos, prompt_ids_lens2, input_ids2, attention_masks2
        else:
            prompt_ids_lens = []
            input_ids = []
            attention_masks = []
            infos = []

            for prompt_ids_len, input_id, attention_mask, info in item_list:
                prompt_ids_lens.append(prompt_ids_len)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                infos.append(info)

            input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)

            attention_masks = zero_pad_sequences(attention_masks, "right")

            return prompt_ids_lens, input_ids, attention_masks, infos
