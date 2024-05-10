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
idx = 0


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
        for e in splits[cfg.alfred.eval_set]:       # e : {'repeat_idx': 2, 'task': 'pick_two_obj_and_place-SoapBar-None-GarbageCan-418/trial_T20190909_055649_717880'} 이런 형식을 가지고있음
            if 'pick_two_obj_and_place' not in e['task']:
                # print(e['task'])       # task type 확인 가능
                files.append(e)

        # select a subset of tasks
        if cfg.alfred.eval_portion_in_percent < 100:
            random.seed(1)  # fix seed for reproducibility
            n_sample = int(len(files) * cfg.alfred.eval_portion_in_percent / 100)
            files = random.sample(files, n_sample)
            print('total files num : ', len(files))
            random.seed(cfg.planner.random_seed)

        # find만 겁나많이할때 오류남 

        if False:  # debug mode
            debug_files = ['trial_T20190906_234406_356490'] # trial_T20190907_054906_608944
            # debug_files = ['trial_T20190908_050518_595510_0', 'trial_T20190908_023400_293044_0', 'trial_T20190908_175253_104175_2', 'trial_T20190909_141915_002879_1', 'trial_T20190906_194903_710920_2',
            #                'trial_T20190908_033721_967359_0', 'trial_T20190907_222640_487432_1', 'trial_T20190908_052232_887934_2', 'trial_T20190907_070152_814652_2', 'trial_T20190910_112922_368384_1',
            #                'trial_T20190906_201148_878110_2', 'trial_T20190909_124835_952557_2', 'trial_T20190906_201106_979461_0', 'trial_T20190908_124340_258012_1', 'trial_T20190907_183137_838565_1',
            #                'trial_T20190909_112912_154874_2', 'trial_T20190906_201106_979461_1', 'trial_T20190907_060234_011675_1', 'trial_T20190909_032318_169393_1', 'trial_T20190909_055248_059513_2',
            #                'trial_T20190906_194903_710920_0', 'trial_T20190908_122858_883968_2', 'trial_T20190907_114931_692464_1', 'trial_T20190907_061934_041977_0', 'trial_T20190909_051813_197569_0',
            #                'trial_T20190906_190942_706186_1', 'trial_T20190908_122858_883968_1', 'trial_T20190906_201148_878110_1', 'trial_T20190908_173610_612042_1', 'trial_T20190908_192636_561572_2',
            #                'trial_T20190907_232225_725376_2', 'trial_T20190906_235214_456832_1', 'trial_T20190907_070857_687585_4', 'trial_T20190909_124835_952557_1', 'trial_T20190909_193045_208933_0', 
            #                'trial_T20190909_055248_059513_1', 'trial_T20190908_191121_189259_2', 'trial_T20190909_154212_097094_0', 'trial_T20190908_110131_524518_0', 'trial_T20190908_050518_595510_1',
            #                'trial_T20190908_023400_293044_2', 'trial_T20190909_045605_307949_0', 'trial_T20190907_151802_277016_0', 'trial_T20190906_192823_237997_1', 'trial_T20190907_024634_972453_0', 
            #                'trial_T20190911_131350_027076_0', 'trial_T20190909_064046_163660_0', 'trial_T20190907_183137_838565_2', 'trial_T20190911_045200_839773_0', 'trial_T20190909_064046_163660_5',
            #                'trial_T20190907_041436_588106_2']
            new_files = []
            print('debug files length : ', len(debug_files))
            
            for file in files:
                if any(debug_file in file['task'] for debug_file in debug_files):
                    new_files.append(file)
            print(new_files, ', debug file num : ', len(new_files))
            files = new_files

        # run
        start = time.time()
        x_display = cfg.alfred.x_display
        save_path = cfg.out_dir
        os.mkdir(save_path+'/test')
        results, fail_totals, success_totals = self.evaluate_main(files, args_dict, planner, x_display, save_path)
        # print('before entropy calc fail totals : ', fail_totals)

        # # print entropy
        # result_entropy_fail = []
        # result_entropy_success = []

        # fail_avg = self.evaluate_evg(fail_totals)
        # success_avg = self.evaluate_evg(success_totals)
        # print('fail_average : ', fail_avg)
        # print('total_fail_average : ', np.mean(self.evaluate_evg(fail_totals)))
        # print('total_fail_average_1 : ', np.percentile(self.evaluate_evg(fail_totals), 25))
        # print('total_fail_average_3 : ', np.percentile(self.evaluate_evg(fail_totals), 75))

        # entropy_fail = self.entropy(fail_totals)
        # print('entropy_fail : ', entropy_fail)
        # entropy_success = self.entropy(success_totals)

        # result_entropy_fail = entropy_fail
        # result_entropy_success = entropy_success

        # print('result_entropy_fail average : ', np.mean(result_entropy_fail))
        # print('result_entropy_success average : ', np.mean(result_entropy_success))

        # # save plots
        # _, ax = plt.subplots(nrows=1, figsize = (80, 15))           # 80으로 하니까 task 수 다 나옴

        # ax.set_xlabel('# of Task', fontsize=50)
        # ax.set_ylabel('score difference', fontsize=50)

        # # ax[1].set_xlabel('# of Task', fontsize=15)
        # # ax[1].set_ylabel('score difference value & entropy', fontsize=15)

        # # ax.set_title('probability distribution', fontsize=50)
        # # ax[1].set_title('success distribution & entropy', fontsize=20)

        # sns.lineplot(data=fail_avg, ax=ax, linewidth=4)
        # # sns.lineplot(data=success_avg, ax=ax, color='darkgreen', linewidth=4)
        # # sns.swarmplot(data=success_totals, ax=ax[1])
        # model_wise_threshold = -0.58945
        # plt.axhline(model_wise_threshold, 0, 1, color='darkorange', linestyle='--', linewidth=3)
        
        # # 축 눈금 사이즈
        # plt.xticks(fontsize=35)
        # plt.yticks(fontsize=35)

        # # sns.lineplot(data=result_entropy_fail, ax=ax[0])
        # # sns.lineplot(data=result_entropy_success, ax=ax[1])
        # plt.savefig(os.path.join(save_path, 'score difference.png'))        # 스레드 에러나는데 무시해도 됨ㅇㅇ

        # # print results
        # log.info(results)
        # log.info(f'find diff scores (fail) : {fail_totals}')        # fail task에 대한 diff score 수집
        # log.info(f'find diff scores (success) : {success_totals}')
        
        n = len(results)
        n_success = 0
        for e in results:
            if e['success']:
                n_success += 1
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start)))}')

    def evaluate_evg(self, probs):
        averages = []

        for values in probs:
            values = np.array(values)
            avg = np.mean(values)
            averages.append(avg)
        
        return averages

    def entropy(self, probs):
        entropies = []

        for values in probs:
            values = np.array(values)
            _, counts = np.unique(values, return_counts=True)
            probabilities = counts / len(values)
            entropy = -np.sum(probabilities * np.log2(probabilities[probabilities > 0]))

            entropies.append(entropy)

        return entropies

    def evaluate_main(self, tasks, args_dict, planner, x_display, save_path):
        results = []
        fail_total = []
        success_total = []
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
                r_idx = task['repeat_idx']      # trial_T.....__0 <- 맨마지막 순서가 r_idx (task['repeat_idx'])
                log.info(f"Evaluating ({i+1}/{len(tasks)}): {traj_data['root']}")
                result, total_diff = self.evaluate_task(env, traj_data, r_idx, model_args, planner, save_path, log_prompt=(i==0), train_gt_steps=train_gt_steps, valid_seen_gt_steps=valid_seen_gt_steps)
                if result['success'] is not True:
                    print('not success!! total append ok')
                    fail_total.append(total_diff)
                else:
                    print('success!! total append ok')
                    success_total.append(total_diff)
                results.append(result)

            except Exception as e:
                import traceback
                traceback.print_exc()
                log.info("Error: " + repr(e))

        return results, fail_total, success_total
    
    def evaluate_task(self, env, traj_data, r_idx, model_args, planner, save_path, log_prompt=False, train_gt_steps=None, valid_seen_gt_steps=None):
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
        img_num = 0
        global idx
        seperate_imgs_path = save_path+'/test'  
        imgs = [Image.fromarray(env.last_event.frame)]
    
        prev_steps = []
        prev_action_msg = []
        prev_diff_score = []
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
                step, prompt, diff_score = planner.plan_step_by_step(instruction_text, prev_steps, prev_action_msg)

                if step is None:
                    log.info("\tmax step reached")
                    break
                
                if diff_score is not None:
                    prev_diff_score.append(diff_score)
                    # print('length: ', len(prev_diff_score), ', prev_diff_score : ', prev_diff_score)

            # prompt 로그
            # if log_prompt:
            #     log.info(prompt)

            log.info(f'{len(prev_steps) + 1}. {step}')
            prev_steps.append(step)

            print('Ground Truth :', valid_seen_gt_steps[traj_data['task_id']])
            print('step :', step)

            if t >= len(valid_seen_gt_steps[traj_data['task_id']]):
                print('GT count exceeded')
            elif step == valid_seen_gt_steps[traj_data['task_id']][t]:
                print('match')
            elif step != valid_seen_gt_steps[traj_data['task_id']][t]:
                print('mismatch')
            else:
                print('except')
            
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
            draw_imgs, origin_imgs = env.write_step_on_img(self.cfg.planner.use_predefined_prompt, t+1, action_ret)
            imgs.append(draw_imgs)
            
            origin_imgs.save(os.path.join(seperate_imgs_path, f"{idx}_{img_num}.png"))
            img_num += 1
            print('num: ', img_num, ', ', idx)

            prev_action_msg.append(action_ret['message'])

            if not action_ret['success']:
                print(action_ret['message'])

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1
        
        idx += 1

        # while 끝나면 target goal : {}... success : True or False 출력하는거 나옴
        print('이 task의 최종 length: ', len(prev_diff_score), '  prev_diff_score : ', prev_diff_score)

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
        self.save_result(env, log_entry, imgs, save_path)

        return log_entry, prev_diff_score

    def save_result(self, env, result_dict, imgs, base_path='results'):
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
        textbox_height = 70  # max two lines
        total_height = math.ceil(len(imgs) / 5) * heights[0] + textbox_height
        new_im = Image.new('RGB', (total_width, total_height), color='white')

        # draw text
        if result_dict:
            text = 'Instruction: ' + result_dict['goal_instr']
            text2 = 'goal: ' + json.dumps(env.task.get_targets())
            text_color = (0, 0, 0)  # black
            # text_color = (100, 255, 100) if result_dict['success'] else (255, 100, 100)
            lines = textwrap.wrap(text, width=110)
            lines2 = textwrap.wrap(text2, width=110)
            #lines3 = lines + lines2
            draw = ImageDraw.Draw(new_im)
            y_start = 10 if len(lines) > 1 else 35
            #draw.multiline_text((10, y_start), '\n'.join(lines3), font=font, fill=text_color)
            draw.text((2, 2), text, font=font, fill=text_color)
            draw.text((2, y_start + 5), text2, font=font, fill=text_color)
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
