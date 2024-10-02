```yaml
license: apache-2.0
datasets:
  - Skywork/Skywork-Reward-Preference-80K-v0.1
base_model:
  - Qwen/Qwen2-7B-Instruct
```

## Introduction

Con-J-Qwen2-7B (learning the generative \***J***udge using self-generated ***Con***trastive judgments) is an advanced generative judge built on Qwen2-7B-Instruct architecture and dataset Skywork/Skywork-Reward-Preference-80K-v0.1. 

Con-J-Qwen2-7B is trained from preference data. We prompt the pre-trained Qwen2-7B-Instruct model to generate positive and negative judgments, both supported with rationales in natural language form. Then the self-generated contrastive judgment pairs are used to train the generative judge with Direct Preference Optimization (DPO). By doing this, Con-J learns to act as a generative judge and provides accurate and supprting rationales.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Model/Con-J-Qwen2-7B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

question = "What is the range of the numeric output of a sigmoid node in a neural network?"
answer1 = "The output of a sigmoid node is bounded between -1 and 1."
answer2 = "The output of a sigmoid node is bounded between 0 and 1."

# Format and tokenize the conversations
CON_J_PROMPT = """作为一个评价专家，给定一个问题和它的两个可能的回答，请选出哪一个回答在连贯性、准确性、覆盖度和上述定义的整体质量方面最为符合。请用JSON格式输出你的判断, 其中"原因"是你提供的解释，"更好的回答"是整数类型的1或2，例如{{"原因": "你的解释", "更好的回答": 1}}。以下是问题和候选回答的内容：
    \n问题：{instruction}
回答1：{output_1}
回答2：{output_2}"""
user_prompt = CON_J_PROMPT.format(instruction=question, output_1=answer1, output_2=answer2)
system_prompt = ""
messages = [
    {"role": "system", "content": system_prompt,},
    {"role": "user", "content": user_prompt},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt = tokenizer([prompt], return_tensors="pt")

# Generate judgment for the given prompt
with torch.no_grad():
    generated_ids = model.generate(prompt.input_ids, do_sample=False, max_new_tokens=2048,)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# response: {"原因": "回答1中的-1是错误的，因为sigmoid函数的实际输出范围是0到1，而不是包括-1。回答2准确地描述了sigmoid函数的输出范围是0到1。",\n "更好的回答": 2}

```


## Performance

<table>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Infinity-<br>Preference</th>
    <th rowspan="2">Ultra-<br>Feedback</th>
    <th rowspan="2">PKU-<br>SafeRLHF</th>
    <th colspan="4">Reward-Bench</th>
  </tr>
  <tr>
    <th>Chat</th>
    <th>Chat-H</th>
    <th>Safety</th>
    <th>Reasoning</th>
  </tr>
  <tr>
    <td>Llama3.1-8B</td>
    <td>59.0</td>
    <td>62.9</td>
    <td>66.4</td>
    <td>80.7</td>
    <td>49.8</td>
    <td>64.0</td>
    <td>68.1</td>
  </tr>
  <tr>
    <td>Llama3.1-70B</td>
    <td>64.0</td>
    <td>71.4</td>
    <td>67.6</td>
    <td><b>97.2</b></td>
    <td>70.2</td>
    <td>82.8</td>
    <td>86.0</td>
  </tr>
  <tr>
    <td>Qwen2-7B</td>
    <td>59.0</td>
    <td>64.5</td>
    <td>67.2</td>
    <td>91.3</td>
    <td>44.8</td>
    <td>73.6</td>
    <td>69.0</td>
  </tr>
  <tr>
    <td>Qwen2.5-72B</td>
    <td>70.0</td>
    <td>66.0</td>
    <td>58.7</td>
    <td>86.6</td>
    <td>61.4</td>
    <td>74.5</td>
    <td><b>90.7</b></td>
  </tr>
  <tr>
    <td>Auto-J</td>
    <td>69.0</td>
    <td>63.9</td>
    <td>66.9</td>
    <td>93.0</td>
    <td>40.0</td>
    <td>65.5</td>
    <td>50.5</td>
  </tr>
  <tr>
    <td>Prometheus 2</td>
    <td>68.0</td>
    <td>63.3</td>
    <td>63.0</td>
    <td>85.5</td>
    <td>49.1</td>
    <td>77.1</td>
    <td>76.5</td>
  </tr>
  <tr>
    <td>GPT-4o</td>
    <td><u>75.0</u></td>
    <td><u>72.2</u></td>
    <td><b>69.6</b></td>
    <td><u>95.3</u></td>
    <td><u>74.3</u></td>
    <td><u>87.6</u></td>
    <td>86.9</td>
  </tr>
  <tr>
    <td>Con-J (ours)</td>
    <td><b>81.0</b></td>
    <td><b>73.0</b></td>
    <td><u>68.4</u></td>
    <td>91.3</td>
    <td><b>79.6</b></td>
    <td><b>88.0</b></td>
    <td><u>87.1</u></td>
  </tr>
</table>

## Training Scripts
The training of Con-J is based on a self-modified version of [Open-RLHF](https://github.com/OpenRLHF/OpenRLHF).
The training scripts are available in Code/run_scripts/. The training of Con-J involves the following steps:
```bash
task_name="Skywork-Reward-Preference-80K-v0.1"
cd run_scripts/Qwen2/
# repeated sampling
sh vllm_inference_best_of_n.sh 8 $task_name
# hint driven sampling
sh vllm_inference_all.sh $task_name
# dataset filtering and construction
python ../../examples/construct_dpo_dataset_for_critic_model.py --task $task_name
# contrastive training
sh train_dpo.sh $task_name
# inference and evaluation
sh vllm_inference2.sh $task_name $task_name
```
To enable Con-J training, one should download the base model Qwen/Qwen2-7B-Instruct and the dataset Skywork/Skywork-Reward-Preference-80K-v0.1 to proper place align with the training scripts. Then the downloaded dataset can be preprocessed by runing the following command:
```bash
python preprocess_dataset.py
```

## Reference
Coming soon.

