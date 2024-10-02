import math
from abc import ABC
import torch.nn.functional as F
import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from openrlhf.models import GPTLMLoss
import json
import os
import numpy as np
import copy

class RewardModelTrainer2(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        critic_model,
        strategy,
        optim: Optimizer,
        optim_c,
        train_dataloader,
        eval_dataloader,
        scheduler,
        scheduler_c,
        tokenizer,
        max_norm=0.5,
        max_epochs: int = 2,
        loss="bce",
        mode = 's1',
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.critic_model = critic_model
        self.mode = mode
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.scheduler_c = scheduler_c
        self.optimizer = optim
        self.optimizer_c = optim_c
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.loss = loss
        if loss == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        self.gpt_loss_fn = GPTLMLoss()

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def right_to_left_padding(input_ids, attention_mask, pad_token_id=1):
        max_length = input_ids.size(1)
        left_padded_input_ids = []
        left_padded_attention_mask = []

        for seq_ids, seq_mask in zip(input_ids, attention_mask):
            # 找到非填充部分
            non_pads_ids = seq_ids[seq_mask != 0]
            non_pads_mask = seq_mask[seq_mask != 0]
            
            # 计算左填充的长度
            left_padding_length = max_length - non_pads_ids.size(0)
            
            # 创建左填充序列
            left_padded_ids = torch.cat([torch.tensor([pad_token_id] * left_padding_length), non_pads_ids])
            left_padded_mask = torch.cat([torch.tensor([0] * left_padding_length), non_pads_mask])
            
            left_padded_input_ids.append(left_padded_ids)
            left_padded_attention_mask.append(left_padded_mask)

        return torch.stack(left_padded_input_ids), torch.stack(left_padded_attention_mask)
    
    def step(self, strategy, loss, optimizer, scheduler, model):
        strategy.backward(loss, model, optimizer)
        strategy.optimizer_step(optimizer, model, scheduler)

    def compute_model_list(self, prompt_ids, prompt_mask, chosens, prompts_id_len, mode, show_case=False, return_logits = False, return_reward = False):
        if self.margin_loss:
            margin = torch.tensor(margin).to(torch.cuda.current_device())
        else:
            margin = None
        loss, all_values = 0, None
        
        if 's' in mode:
            all_values, output = self.model(prompt_ids, attention_mask=prompt_mask, prompts_id_len=prompts_id_len, return_output=True, )
            if self.args.output_attentions:
                attentions = torch.stack(output['attentions'])
                attentions = attentions[:,:,:,-1]
                attentions = torch.transpose(attentions, 0, 1)
                avg_attentions = torch.mean(attentions, dim=(1,2))  # [batch_size, tokens]
                logits = {
                    'attentions': avg_attentions.detach().cpu().float().numpy().tolist(),
                    'input_ids': prompt_ids.detach().cpu().numpy().tolist(),
                    'attention_mask': prompt_mask.detach().cpu().numpy().tolist(),
                }
            elif return_logits:
                logits = output['last_hidden_state'].squeeze(-1).detach().cpu().float().numpy()[:,-1,:].tolist()
            if self.compute_fp32_loss:
                all_values = all_values.float()

            if self.loss == 'bce':
                loss = self.loss_fn(all_values, chosens)
            else:
                loss = self.loss_fn(all_values, chosens.long())
           
        if 'c' == mode:
            labels = torch.where(prompt_mask.bool(), prompt_ids, self.gpt_loss_fn.IGNORE_INDEX,)
            for label, source_len in zip(labels, prompts_id_len):
                label[:source_len] = self.gpt_loss_fn.IGNORE_INDEX
            output = self.critic_model(prompt_ids, attention_mask=prompt_mask, return_output=True)
            loss = self.gpt_loss_fn(output.logits, labels)

        if not self.aux_loss:
            aux_loss = 0
        loss += aux_loss * self.args.aux_loss_coef
        if 'new' in self.args.mode:
            logits = [output['reward1'], output['reward2']]
        if return_logits or self.args.output_attentions or 'new' in self.args.mode:
            return loss, all_values, aux_loss, logits
        else:
            return loss, all_values, aux_loss

    def get_acc(self, all_values, chosens, split=False):
        if split:
            if self.loss == 'bce':
                probs = (torch.sigmoid(all_values) >= 0.5).float()
                return (probs == (chosens)).int()
            else:
                return (torch.max(all_values, 1)[1] == (chosens)).int()
        else:
            if self.loss == 'bce':
                probs = (torch.sigmoid(all_values) >= 0.5).float()
                return (probs == (chosens)).sum().item() / len(chosens)
            else:
                return (torch.max(all_values, 1)[1] == (chosens)).sum().item() / len(chosens)

    def fit(self, args):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        
        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            acc_mean = 0.5
            loss_mean = 0

            for prompt_ids, prompt_mask, chosens, extra, meta_info, prompts_id_len in self.train_dataloader:
                mode = args.mode
                if mode =='c':
                    self.critic_model.train()
                if 's' in mode:
                    self.model.train()
                        
                prompt_ids = prompt_ids.squeeze(1).to(torch.cuda.current_device())
                prompt_mask = prompt_mask.squeeze(1).to(torch.cuda.current_device())
                chosens = chosens.to(torch.cuda.current_device())
                
                loss, all_values, aux_loss = self.compute_model_list(prompt_ids, prompt_mask, chosens, prompts_id_len, mode, show_case= True if global_step % 20 == 1 else False, return_reward=True if 'new' in args.mode else False)
                if 'c' == mode:
                    self.step(self.strategy, loss, self.optimizer_c, self.scheduler_c, self.critic_model)      
                if 's' in mode:
                    self.step(self.strategy, loss, self.optimizer, self.scheduler, self.model)
                acc = self.get_acc(all_values, chosens)
                acc_mean = acc_mean * 0.9 + 0.1 * acc

                loss = loss.item()
                loss_mean = loss_mean * 0.9 + 0.1 * loss
                # optional rm info
                logs_dict = {
                    "loss": loss,
                    "acc_mean": acc_mean,
                    'acc': acc,
                    "loss_mean": loss_mean,
                }

                # logs/checkpoints/evaluate
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if self.args.mode == 'c':
                self.strategy.save_model(self.critic_model, self.tokenizer, os.path.join(args.ckpt_path, tag, 'c'))
            if 's' in self.args.mode:
                self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.ckpt_path, tag, self.args.mode))

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        with torch.no_grad():
            acc, loss_sum = 0, 0
            mode = self.args.mode
            if mode =='c':
                self.critic_model.eval()
            if 's' in mode:
                self.model.eval()
            for prompt_ids, prompt_mask, chosens, extra, meta_info, prompt_ids_len in eval_dataloader:
                prompt_ids = prompt_ids.squeeze(1).to(torch.cuda.current_device())
                prompt_mask = prompt_mask.squeeze(1).to(torch.cuda.current_device())
                chosens = chosens.to(torch.cuda.current_device())
                loss, all_values, aux_loss = self.compute_model_list(prompt_ids, prompt_mask, chosens, prompt_ids_len, mode=mode)
                if all_values is not None:
                    # acc += ((torch.abs(all_values - chosens) < 0.5).int()).float().mean().item()
                    acc += self.get_acc(all_values, chosens)
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc / self.eval_dataloader.__len__()
            loss_mean = loss_sum / self.eval_dataloader.__len__()
            
            if all_values is not None:
                all_values = self.strategy.all_gather(all_values)
                reward_mean = torch.mean(all_values)
                reward_std = torch.std(all_values).clamp(min=1e-8)
                self.strategy.print("Set reward mean std")
                model = self.model
                unwrap_model = self.strategy._unwrap_model(model)
                unwrap_model.config.mean = reward_mean.item()
                unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)

    def test(self, eval_dataloader, get_token_info = False, output_file=None):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Test stage...",
            disable=not self.strategy.is_rank_0(),
        )
        all_info_dict_mode = {'loss':[], 'acc':[], 'tags':[], 'test_id': [], 'logits':[], 'scores':[], 'chosens':[]}
        mode = self.args.mode
        if mode =='c':
            self.critic_model.eval()
        if 's' in mode:
            self.model.eval()
        with torch.no_grad():
            acc, acc_list = 0, []
            tags = []
            loss_sum = 0
            for prompt_ids, prompt_mask, chosens, extra, meta_infos, prompt_ids_len in eval_dataloader:
                prompt_ids = prompt_ids.squeeze(1).to(torch.cuda.current_device())
                prompt_mask = prompt_mask.squeeze(1).to(torch.cuda.current_device())
                chosens = chosens.to(torch.cuda.current_device())
                if self.args.output_attentions or 'new' in mode:
                    loss, all_values, aux_loss, logits = self.compute_model_list(prompt_ids, prompt_mask, chosens, prompt_ids_len, mode=mode,)                    
                elif get_token_info:
                    loss, all_values, aux_loss, logits = self.compute_model_list(prompt_ids, prompt_mask, chosens, prompt_ids_len, mode=mode, return_logits=True)
                else:
                    loss, all_values, aux_loss = self.compute_model_list(prompt_ids, prompt_mask, chosens, prompt_ids_len, mode=mode)

                all_info_dict = copy.deepcopy(all_info_dict_mode)
                if all_values is not None:
                    acc_bool1 = self.get_acc(all_values, chosens, split = True)
                    # acc_bool1 = (torch.max(all_values, 1)[1] == (chosens)).int()
                    acc += acc_bool1.float().mean().item() 
                    acc_list.extend(acc_bool1.float().flatten().tolist())
                    all_info_dict['acc'] = (acc_bool1.flatten().tolist())

                tags.extend([meta["tag"] for meta in meta_infos])

                loss_sum += loss.mean().item()
                step_bar.update()
                if get_token_info or self.args.output_attentions: 
                    all_info_dict['logits'] = logits
                    all_info_dict['chosens'] = chosens.detach().cpu().tolist()
                if 'new' in mode:
                    all_info_dict['logits'] = logits
                all_info_dict['scores'] = all_values.detach().cpu().tolist()
                all_info_dict['tag'] = ([meta["tag"] for meta in meta_infos])
                all_info_dict['test_id'] = ([meta["test_id"] for meta in meta_infos])
                all_info_dict['loss'] = (loss.detach().cpu().tolist()) 
                output_file.write(json.dumps(all_info_dict,) + '\n')
                output_file.flush()
 
            self.strategy.print(f"Test Data Size: {eval_dataloader.__len__()}")
            acc_mean = acc / eval_dataloader.__len__()
            loss_mean = loss_sum / eval_dataloader.__len__()

            bar_dict = {
                f"eval_loss": loss_mean,
                f"acc_mean": acc_mean,
            }
            self.strategy.print(bar_dict)
            from collections import defaultdict
            self.strategy.print(f"===========Detail ACC==========")
            acc_dict = defaultdict(list)
            overall_acc = []
            for tag, acc in zip(tags, acc_list):
                acc_dict[tag].append(acc)
                overall_acc.append(acc)

            sort_keys = sorted(list(acc_dict.keys()))
            for tag in sort_keys:
                accs = acc_dict[tag]
                self.strategy.print(f"{tag}:\t\t{np.mean(accs):.4f}\t\tdata length:{len(accs)}")

            self.strategy.print(f"Overall:\t\t{np.mean(overall_acc):.4f}")

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, return_all = False):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)

        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        if return_all:
            return chosen_rewards, rejected_rewards, aux_loss, output
        else:
            return chosen_rewards, rejected_rewards, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks
