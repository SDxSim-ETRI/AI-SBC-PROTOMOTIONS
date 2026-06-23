import sys, argparse
from copy import deepcopy
from pathlib import Path
import torch

sys.path.insert(0, str(Path('.').resolve()))

CHECKPOINT_DIR = Path('data/pretrained_models/motion_tracker/g1-bones-deploy')

def load_experiment_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('experiment_config', CHECKPOINT_DIR / 'experiment_config.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def make_args():
    args = argparse.Namespace()
    args.robot_name = 'g1'
    args.simulator = 'mujoco'
    args.headless = True
    args.num_envs = 1
    args.experiment_name = 'g1-bones-deploy'
    args.batch_size = 4096
    args.training_max_steps = 1_000_000_000
    args.motion_file = 'data/motion_for_trackers/g1_bones_seed_mini.pt'
    args.scenes_file = None
    args.use_wandb = False
    return args

def main():
    print('Loading experiment module...')
    exp = load_experiment_module()
    args = make_args()

    from protomotions.robot_configs.factory import robot_config as robot_config_factory
    from protomotions.simulator.factory import simulator_config as simulator_config_factory

    print('Building configs...')
    robot_cfg = robot_config_factory(args.robot_name)
    simulator_cfg = simulator_config_factory(args.simulator, robot_cfg, args.headless, args.num_envs, args.experiment_name)
    exp.configure_robot_and_simulator(robot_cfg, simulator_cfg, args)
    terrain_cfg = exp.terrain_config(args)
    scene_lib_cfg = exp.scene_lib_config(args)
    motion_lib_cfg = exp.motion_lib_config(args)
    env_cfg = exp.env_config(robot_cfg, args)
    agent_cfg = exp.agent_config(robot_cfg, env_cfg, args)

    training_configs = dict(robot=robot_cfg, simulator=simulator_cfg, terrain=terrain_cfg, scene_lib=scene_lib_cfg, motion_lib=motion_lib_cfg, env=env_cfg, agent=agent_cfg)
    torch.save(training_configs, CHECKPOINT_DIR / 'resolved_configs.pt')
    print('Saved resolved_configs.pt')

    from protomotions.utils.inference_utils import apply_all_inference_overrides
    robot_cfg_inf = deepcopy(robot_cfg)
    simulator_cfg_inf = deepcopy(simulator_cfg)
    terrain_cfg_inf = deepcopy(terrain_cfg)
    scene_lib_cfg_inf = deepcopy(scene_lib_cfg)
    motion_lib_cfg_inf = deepcopy(motion_lib_cfg)
    env_cfg_inf = deepcopy(env_cfg)
    agent_cfg_inf = deepcopy(agent_cfg)
    apply_all_inference_overrides(robot_cfg_inf, simulator_cfg_inf, env_cfg_inf, agent_cfg_inf, terrain_cfg_inf, motion_lib_cfg_inf, scene_lib_cfg_inf, experiment_module=exp, args=args)
    inference_configs = dict(robot=robot_cfg_inf, simulator=simulator_cfg_inf, terrain=terrain_cfg_inf, scene_lib=scene_lib_cfg_inf, motion_lib=motion_lib_cfg_inf, env=env_cfg_inf, agent=agent_cfg_inf)
    torch.save(inference_configs, CHECKPOINT_DIR / 'resolved_configs_inference.pt')
    print('Saved resolved_configs_inference.pt')
    print('Done!')

if __name__ == '__main__':
    main()
