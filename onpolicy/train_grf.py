import os
import sys
import wandb
import torch
import socket
import setproctitle

import numpy as np

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './envs/package/')))

from config import get_config

from pathlib import Path


from envs.football.Football_Environment import FootballEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from runner.shared.football_runner import FootballRunner as Runner

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Football":
                env = FootballEnv(all_args)
 
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Football":
                env = FootballEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default="curriculum_learning",
                        choices = [
                            "1_vs_1_easy",
                            "curriculum_learning_11vs11",
                            "curriculum_learning",
                            "11_vs_11_easy_stochastic", 
                            "11_vs_11_stochastic",
                            "11_vs_11_hard_stochastic",
                        ], 
                        help="which scenario to run on.")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="number of controlled players. (must >= 3)")
    parser.add_argument("--representation", type=str, default="simple115v2", 
                        choices=["simple115v2", "extracted", "pixels_gray", 
                                 "pixels"],
                        help="representation used to build the observation.")
    parser.add_argument("--rewards", type=str, default="scoring", 
                        help="comma separated list of rewards to be added.")
    parser.add_argument("--smm_width", type=int, default=96,
                        help="width of super minimap.")
    parser.add_argument("--smm_height", type=int, default=72,
                        help="height of super minimap.")
    parser.add_argument("--remove_redundancy", action="store_true", 
                        default=False, 
                        help="by default False. If True, remove redundancy features")
    parser.add_argument("--zero_feature", action="store_true", 
                        default=False, 
                        help="by default False. If True, replace -1 by 0")
    parser.add_argument("--eval_deterministic", action="store_false", 
                        default=True, 
                        help="by default True. If False, sample action according to probability")
    parser.add_argument("--share_reward", action='store_false', 
                        default=True, 
                        help="by default true. If false, use different reward for each agent.")
    parser.add_argument("--save_videos", action="store_true", default  = False, 
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_dir", type=str, default="./render", 
                        help="directory to save videos.")
                        
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
        
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
        all_args.use_joint_action_loss = False


    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
        all_args.use_joint_action_loss = False

    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    elif all_args.algorithm_name == "jrpo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
        all_args.use_joint_action_loss = True
        
    elif all_args.algorithm_name == "tizero":
        if "11_vs_11" not in all_args.scenario_name and all_args.representation != "simple115v2" :
            raise ValueError("Tizero는 11 vs 11 simple115v2 관찰 환경에서만 실행가능합니다.")
        print("u are choosing to use Tizero, we set use_joint_action_loss to be True")
        all_args.use_joint_action_loss = True
    else:
        raise NotImplementedError

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(f"cuda:{all_args.num_gpu}")
        torch.set_num_threads(all_args.n_torch_threads)
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_torch_threads)
    
    # fix seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    """
    실험 기록을 저장할 디렉토리 생성
    """
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=all_args.notes, 
                         name="-".join([
                            all_args.algorithm_name,
                            all_args.experiment_name,
                            "seed" + str(all_args.seed)
                         ]),
                         group=all_args.group_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # 프로세스의 제목을 결정합니다. 
    setproctitle.setproctitle("-".join([
        all_args.env_name, 
        all_args.scenario_name, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]) + "@" + all_args.user_name)
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    print(f"사용 가능한 CPU Thread: {os.cpu_count()}")

    config = {
        "all_args": all_args,
        "device": device,
        "run_dir": run_dir
    }

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(args = sys.argv[1:])