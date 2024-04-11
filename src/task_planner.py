import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from torch.nn import CrossEntropyLoss
import guidance
import logging
from matplotlib import pyplot as plt
import seaborn as sns 
import math

log = logging.getLogger(__name__)

class TaskPlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.planner.device
        self.max_steps = cfg.planner.max_steps
        self.model_name = cfg.planner.model_name
        self.scoring_batch_size = cfg.planner.scoring_batch_size
        self.score_function = cfg.planner.score_function
        self.scoring_mode = cfg.planner.scoring_mode
        self.use_predefined_prompt = cfg.planner.use_predefined_prompt
        self.thresholding = cfg.planner.thresholding

        # Load pre-trained model
        print(f"Loading LLM and tokenizer: {self.model_name}")

        model_args = {'pretrained_model_name_or_path': self.model_name, 'trust_remote_code': True,
                      'torch_dtype': torch.float16}
        if cfg.planner.use_accelerate_device_map:
            model_args['device_map'] = "auto"
            if cfg.planner.load_in_8bit:
                model_args['load_in_8bit'] = True
        model_args['use_auth_token'] = cfg.planner.hf_auth_token

        if cfg.planner.scoring_mode == 'guidance':
            model_args.pop('pretrained_model_name_or_path')
            tokenizer = None
            if "OpenAI" in self.model_name:
                openai_model_name = self.model_name.split('/')[1]
                guidance.llm = guidance.llms.OpenAI(openai_model_name, api_key=cfg.planner.openai_api_key)
            else:
                if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name:
                    tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
                if "bigscience/bloom" == self.model_name:  # bloom 175B
                    model_args['max_memory'] = {0: '60GB', 1: '80GB', 2: '48GB', 3: '48GB', 4: '48GB'}
                guidance.llm = guidance.llms.Transformers(self.model_name, tokenizer=tokenizer, **model_args)
            self.guidance_program = guidance("""{{prompt}} {{select 'step' options=candidates logprobs='score'}}""")

            self.model = None
            self.tokenizer = None

            logging.getLogger("guidance").setLevel(logging.WARNING)

        else:
            if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name:  # these do not work well with automodel
                self.model = LlamaForCausalLM.from_pretrained(**model_args)
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(**model_args)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if not cfg.planner.use_accelerate_device_map:
                self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer.pad_token_id = 0
            print(f"Loading done\n")

        # Load prompt
        self.prompt = self.init_prompt(cfg)

    def reset(self, nl_act_list, nl_obj_list):
        self.nl_obj_list = nl_obj_list
        self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)

    def reset(self):
        self.skill_set = self.init_skill_set()

    def init_prompt(self, cfg):
        raise NotImplementedError()

    def init_skill_set(self, nl_act_list, nl_obj_list):
        raise NotImplementedError()

    def update_skill_set(self, previous_step, nl_obj_list):
        raise NotImplementedError()

    def score(self, prompt, skill_set):
        scores = {}
        batch_skill_set_list = [skill_set[chunk:chunk + self.scoring_batch_size] for chunk in
                                range(0, len(skill_set), self.scoring_batch_size)]

        if self.scoring_mode == 'guidance':                  
            out = self.guidance_program(prompt=prompt, candidates=skill_set)
            scores = out['score']

        elif self.scoring_mode == 'reuse_prompt' or self.scoring_mode == 'naive':
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True)
            if not self.cfg.planner.use_accelerate_device_map:
                prompt_tokens = prompt_tokens.to(self.device)
            prompt_len = prompt_tokens.attention_mask[0].sum().item()

            for batch_skill_set in batch_skill_set_list:
                batch_sentence = [f"{prompt} {skill}" for skill in batch_skill_set]
                size_B = len(batch_skill_set)
                if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name:
                    batch_skill_set_for_model = batch_skill_set
                else:
                    batch_skill_set_for_model = [f" {skill}" for skill in batch_skill_set]

                with torch.no_grad():
                    if self.scoring_mode == 'reuse_prompt':
                            prompt_output = self.model(**prompt_tokens, use_cache=True)
                            skill_tokens = self.tokenizer(batch_skill_set_for_model, add_special_tokens=False,
                                                          return_tensors="pt", padding=True)
                            if not self.cfg.planner.use_accelerate_device_map:
                                skill_tokens = skill_tokens.to(self.device)
                            concat_attention_mask = torch.cat(
                                (prompt_tokens.attention_mask.repeat(size_B, 1), skill_tokens.attention_mask), dim=1)
                            batch_past_key_values = self.duplicate_past_key_values(prompt_output.past_key_values, size_B)
                    elif self.scoring_mode == 'naive':

                            output = self.model(input_ids=skill_tokens.input_ids,
                                                attention_mask=concat_attention_mask,
                                                past_key_values=batch_past_key_values,
                                                return_dict=True)
                            prompt_last_logits = prompt_output.logits[:, -1:, :].repeat(size_B, 1, 1)  # [B, 1, C]
                            logits = torch.cat((prompt_last_logits, output.logits[:, :-1, :]), dim=1)
                            labels = skill_tokens.input_ids
                            attention_mask = skill_tokens.attention_mask
                    elif self.scoring_mode == 'naive':
                        with torch.no_grad():
                            sentence_tokens = self.tokenizer(batch_sentence, add_special_tokens=False, return_tensors="pt",
                                                             padding=True)
                            if not self.cfg.planner.use_accelerate_device_map:
                                sentence_tokens = sentence_tokens.to(self.device)
                            output = self.model(sentence_tokens.input_ids, attention_mask=sentence_tokens.attention_mask,
                                                return_dict=True)
                            print(output)
                            logits = output.logits[:, prompt_len - 1:-1]
                            labels = sentence_tokens.input_ids[:, prompt_len:]
                            attention_mask = sentence_tokens.attention_mask[:, prompt_len:]

                    size_B, size_L, size_C = logits.shape
                    logits = logits.reshape([size_B * size_L, size_C])
                    labels = labels.reshape([size_B * size_L])
                    loss_fn = CrossEntropyLoss(reduction='none')
                    loss = loss_fn(logits.float(), labels.long())
                    loss = loss.reshape([size_B, size_L])
                    skill_len = attention_mask.count_nonzero(axis=1)
                    if self.score_function == 'sum':
                        score = -(loss * attention_mask).sum(axis=1)
                    elif self.score_function == 'avg':
                        score = -(loss * attention_mask).sum(axis=1) / skill_len

                    for skill_id, skill in enumerate(batch_skill_set):
                        scores[skill] = score[skill_id].item()
        else:
            assert False, 'unknown scoring mode'
        return scores
    
    # beam search
    def beam_search(self, query, k):
        sequences = [[list(), 0.0]]

        for row in query:
            all_candidates = list()
        
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                new_seq = seq + [j]
                new_score = score + -log(row[j])
                candidate = [new_seq, new_score]
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:k]

        return sequences

    def plan(self, query):      # 사용 X
        step_seq = []
        skill_set_size_seq = []
        prompt = self.prompt + f'Human: {query}\nRobot: 1.'
        print(f"Input query: {query}")

        # Scoring
        for step in range(self.max_steps):
            # Make batch
            skill_set = self.skill_set

            # Save skill set size
            skill_set_size_seq.append(len(skill_set))

            # Scoring
            scores = self.score(prompt, skill_set)

            # Select best skill
            results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_step = results[0][0]
            step_seq.append(best_step.strip())
            print(f'{step + 1}. {best_step}')

            # Stop criteria
            if best_step in ['done', 'done.', 'done.\n']:
                prompt += f" {best_step}."
                break

            # Update skill set
            self.update_skill_set(best_step, self.nl_obj_list)

            # Update prompt
            prompt += f" {best_step}, {step + 2}."

        return step_seq, skill_set_size_seq

    def plan_step_by_step(self, query, prev_steps=(), prev_msgs=()):
        if len(prev_steps) >= self.max_steps:
            return None, None, None

        prompt = self.prompt + f'Human: {query.strip()}\nRobot: 1. '

        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_predefined_prompt and len(msg) > 0:
                prompt += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                prompt += step + f', {i + 2}. '

        # score
        scores = self.score(prompt, self.skill_set)     # 각 스킬의 score, 가이던스의 score가 출력됨, 딕셔너리

        # find the best step
        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        best_step = results[0][0].strip()
        best_step_prob = results[0][1]
        # second_step = results[1][0].strip()

        # diff = None
        # diff_ex = None
        # if best_step.startswith("find"):
        #     best_step_probs = results[0][1]
        #     #print('best_step_probs : ', best_step_probs)
        #     best_probs = math.exp(best_step_probs)
        #     #print('best_probs (exponential) : ', best_probs)
            
        #     second_step_probs = results[1][1]     # numpy.float64
        #     second_probs = math.exp(second_step_probs)
        #     #print('second_probs (exponential) : ', second_probs)
            
        #     diff = second_step_probs - best_step_probs  # 2순위와 1순위의 score 차이
        #     diff_ex = second_probs - best_probs
        #     #print('diff : ', diff, '\tdiff_ex : ', diff_ex)

        #     # 특정 threshold 이상의 score 차이가 나면 2순위 스킬셋을 선택
        #     # gpt-3.5-turbo-instruct : -0.531865
        #     # davinci-003 : -0.58945
        #     # llama-2-7b : -0.18834
        #     # llama-2-13b : -0.2504
        #     # llama-2-70b : -0.22253
        #     # llama-1-7b : -0.18208
        #     # llama-1-13b : -0.26045
        #     # llama-1-65b : -0.28849
        #     if self.thresholding:
        #             threshold = float(-0.58945)
        #             if diff_ex > threshold:
        #                 best_step = second_step
        #                 print('second : ', best_step)
       
        return best_step, prompt, best_step_prob #diff_ex

    def duplicate_past_key_values(self, past_key_values, batch_size):
        batch_past_key_values = []
        for layer in range(len(past_key_values)):
            batch_past_key_values_layer = []
            for kv in range(len(past_key_values[layer])):
                batch_past_key_values_layer.append(past_key_values[layer][kv].repeat(batch_size, 1, 1, 1))
            batch_past_key_values_layer = tuple(batch_past_key_values_layer)
            batch_past_key_values.append(batch_past_key_values_layer)
        batch_past_key_values = tuple(batch_past_key_values)
        return batch_past_key_values
