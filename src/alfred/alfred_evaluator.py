import math
import os, json
import pprint
import random
import textwrap
import time, datetime
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from PIL import Image, ImageDraw, ImageFont

from alfred.data.preprocess import Dataset
from src.alfred.alfred_task_planner import AlfredTaskPlanner
from src.alfred.thor_connector import ThorConnector
from src.task_planner import TaskPlanner
from src.alfred.utils import dotdict, load_task_json
from tqdm import tqdm
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns 
import numpy as np

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from src.evaluator import Evaluator


splits = 'alfred/data/splits/oct21.json'
font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 24)
log = logging.getLogger(__name__)


class AlfredEvaluator(Evaluator):
    def __init__(self, hparams):
        self.cfg = hparams

    def evaluate(self):
        cfg = self.cfg

        log.info(OmegaConf.to_yaml(cfg))                 # src.alfred.alfred_evaluator][INFO] - planner:
        global splits

        # llm planner
        if len(cfg.planner.model_name) > 0:
            planner = AlfredTaskPlanner(cfg)           
            planner.reset()  # init skill set
        else:
            planner = None

        # prepare
        args_dict = {'data': 'alfred/data/json_2.1.0', 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': 'alfred/models/config/rewards.json', 'max_steps': 1000}

        with open(splits) as f:
            splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in splits.items()})

        # preprocessing
        number_of_dirs = len(list(os.listdir(args_dict['data'])))
        do_preprocessing = number_of_dirs < 50  # one-time process
        if do_preprocessing:
            log.info("\nPreprocessing dataset... Do this once as required:")
            vocab = None  # todo
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(splits)

        # load tasks
        assert cfg.alfred.eval_set in splits.keys()
        files = []

        # exclude two obj task
        for e in splits[cfg.alfred.eval_set]:
            if 'pick_two_obj_and_place' not in e['task']:
                files.append(e)

        # select a subset of tasks
        if cfg.alfred.eval_portion_in_percent < 100:
            random.seed(1)  # fix seed for reproducibility
            n_sample = int(len(files) * cfg.alfred.eval_portion_in_percent / 100)
            files = random.sample(files, n_sample)
            print('total files num : ', len(files))
            random.seed(cfg.planner.random_seed)

        if cfg.planner.debug_mode:  # debug mode
            debug_files = ['trial_T20190907_151802_277016']
            new_files = []
            
            for file in files:
                if any(debug_file in file['task'] for debug_file in debug_files):
                    new_files.append(file)
            files = new_files

        # run
        start = time.time()
        x_display = cfg.alfred.x_display
        save_path = cfg.out_dir
        nonconformity_score = []
        results, true_options, plan_success_rate = self.evaluate_main(files, args_dict, planner, x_display, save_path)

        data_num = 0

        for data in true_options:
            data_num += len(data)
            if len(data) > 0:
                true_step_smx = np.max(data)
                nonconformity_score.append(1 - true_step_smx)

        # ----------------------------------------------------------------------
        n = len(results)
        n_success = 0
        for e in results:
            if e['success']:
                n_success += 1
        
        log.info(f'real success rate : {(np.sum(plan_success_rate) / len(plan_success_rate)):.2f} %')
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start)))}')

    def evaluate_main(self, tasks, args_dict, planner, x_display, save_path):
        results = []
        true_steps = []
        plan_success_rate_list = []
        model_args = dotdict(args_dict)
        env = ThorConnector(x_display=x_display)
        
        # save prompt
        if planner is not None:
            with open(os.path.join(save_path, 'prompt.txt'), 'w') as f:
                f.write(planner.prompt)

        train_gt_steps = None
        if self.cfg.alfred.eval_set == 'train':
            # load ground-truth trajectories
            train_gt_steps = {}
            with open(self.cfg.prompt.example_file_path, 'r') as f:
                train_examples = json.load(f)
            for ex in train_examples:
                train_gt_steps[ex['task id']] = ex['NL steps']

        valid_seen_gt_steps = None
        if self.cfg.alfred.eval_set == 'valid_seen':
            # load ground-truth trajectories
            valid_seen_gt_steps = {}
            with open(self.cfg.prompt.example_file_path, 'r') as f:
                valid_seen_examples = json.load(f)
            for ex in valid_seen_examples:
                valid_seen_gt_steps[ex['task id']] = ex['NL steps']

        # run
        for i, task in enumerate(tqdm(tasks)):
            try:
                log.info(task)
                traj_data = load_task_json(task)
                r_idx = task['repeat_idx']
                idx = i + 1
                log.info(f"Evaluating ({idx}/{len(tasks)}): {traj_data['root']}")
                result, true_options, step_num = self.evaluate_task(env, traj_data, r_idx, idx, model_args, planner, save_path, log_prompt=(i==0), train_gt_steps=train_gt_steps, valid_seen_gt_steps=valid_seen_gt_steps)

                plan_success_rate = len(true_options) / step_num * 100
                plan_success_rate_list.append(plan_success_rate)

                true_steps.append(list(true_options))
                results.append(result)

            except Exception as e:
                import traceback
                traceback.print_exc()
                log.info("Error: " + repr(e))

        return results, true_steps, plan_success_rate_list
    
    def evaluate_task(self, env, traj_data, r_idx, idx, model_args, planner, save_path, log_prompt=False, train_gt_steps=None, valid_seen_gt_steps=None):
        # setup scene
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize
        env.step(dict(traj_data['scene']['init_action']))
        env.set_task(traj_data, model_args, reward_type='dense')

        # print goal instr
        instruction_text = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        log.info("Task: %s" % instruction_text)

        # inference steps from instruction
        done, success = False, False
        t = 0
        reward = 0
        imgs = [Image.fromarray(env.last_event.frame)]

        prev_steps = []
        prev_action_msg = []
        true_options = []

        ground_truth = valid_seen_gt_steps[traj_data['task_id']]

        step_idx = 0
        GT_idx = 0
        GT_arr = [-1] * len(ground_truth)
        while not done:
            if self.cfg.alfred.eval_set == 'train' and train_gt_steps is not None:
                # sanity check for thor connector: use ground-truth steps
                if t >= len(train_gt_steps[traj_data['task_id']]):
                    step = 'done'
                else:
                    step = train_gt_steps[traj_data['task_id']][t]
                prompt = ''
            else:
                # find next step
                step, prompt, prob = planner.plan_step_by_step(instruction_text, prev_steps, prev_action_msg)

                if step is None:
                    log.info("\tmax step reached")
                    break
            
            log.info('-------------------')
            log.info(f'{len(prev_steps) + 1}. {step}')
            prev_steps.append(step)
            step_prob_pair = {'step': step, 'prob': prob}

            # corresponding number compared to the ground truth
            # If there is a duplicate, the first planned step is numbered.
            step_idx += 1
            if not step in ['done', 'done.', 'done.\n']:
                if ground_truth[GT_idx] == step_prob_pair['step']:
                    GT_arr[GT_idx] = step_idx
                    GT_idx += 1
                    true_options.append(step_prob_pair['prob'])

            if step in ['done', 'done.', 'done.\n']:
                done = True
                prev_action_msg.append('')
                break

            # execute
            step_to_execute = step
            try:
                action_ret = env.llm_skill_interact(step_to_execute)
            except Exception as e:
                log.warning(e)
            imgs.append(env.write_step_on_img(self.cfg, t+1, action_ret, GT_arr, idx))
            prev_action_msg.append(action_ret['message'])

            if not action_ret['success']:
                print(action_ret['message'])

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1  # step num

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        log.info('target goal: ' + json.dumps(env.task.get_targets()))
        log.info('success: ' + str(goal_satisfied))
        if goal_satisfied:
            # print("Goal Reached")
            success = True
            # exit()

        # record results
        log_entry = {'trial': traj_data['task_id'],
                    'scene': scene_name,
                    'type': traj_data['task_type'],
                    'repeat_idx': int(r_idx),
                    'goal_instr': instruction_text,
                    'inferred_steps': prev_steps,
                    'success': success}

        # save
        self.save_result(ground_truth, GT_arr, env, log_entry, imgs, save_path)

        return log_entry, true_options, t
    
    def save_result(self, GTstep, GTarr, env, result_dict, imgs, base_path='results'):
        if result_dict:
            filename = f"{result_dict['trial']}_{result_dict['repeat_idx']}"

            # write json
            with open(os.path.join(base_path, filename + '.json'), "w") as outfile:
                json.dump(result_dict, outfile)
        else:
            filename = "images"

        # write img
        widths, heights = zip(*(i.size for i in imgs))
        total_width = widths[0] * 5
        textbox_height = 160  # max six lines
        total_height = math.ceil(len(imgs) / 5) * heights[0] + textbox_height
        new_im = Image.new('RGB', (total_width, total_height), color='white')

        # draw text
        if result_dict:
            text = 'Instruction: ' + result_dict['goal_instr']
            text_gt_arr = '\nGTarr: ' + ','.join(str(s) for s in GTarr)
            text_gt = ', '.join(GTstep)
            text_color = (0, 0, 0)  # black
            draw = ImageDraw.Draw(new_im)
            draw.text((2, 2), text + text_gt_arr, font=font, fill=text_color)
            lines = textwrap.wrap(text_gt, width=120)
            y_start = 50 if len(lines) > 1 else 70
            for line in lines:
                draw.text((2, y_start + 3), line, font=font, fill=text_color)
                y_start += 25
            y_offset = textbox_height
        else:
            y_offset = 0

        x_offset = 0
        for im in imgs:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
            if x_offset >= total_width:
                x_offset = 0
                y_offset += im.size[1]

        success_str = 'success' if result_dict['success'] else 'fail'
        new_im.save(os.path.join(base_path, f"{filename}_{success_str}.png"))
