"""Evaluate a v2 checkpoint with the exact action interface used to train it."""
import argparse
import csv
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform

import torch
from stable_baselines3 import PPO
from experiment import HERE, digest, evaluate


def load_run(path):
    path=Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}. Pass --model with an existing v2 checkpoint.')
    metadata_path=path.parent/'config.json'
    if not metadata_path.is_file():
        metadata_path=path.parent/'summary.json'
    metadata=json.loads(metadata_path.read_text())
    if metadata.get('model_sha256') and digest(path)!=metadata['model_sha256']:
        raise RuntimeError('Selected model checksum does not match config.json')
    config=metadata['environment']
    expected={'alpha','stance_weight','drift_weight','rate_weight','joint_velocity_weight'}
    if set(config)!=expected:
        raise ValueError(f'Unexpected environment settings in {metadata_path}')
    torch.set_num_threads(1)
    model=PPO.load(str(path),device='cpu')
    if model.observation_space.shape != (72,):
        raise ValueError('This evaluator requires a v2 policy with 72 observations.')
    return model,config,path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(HERE/'selected/model.zip'))
    p.add_argument('--seconds',type=float,default=180.0)
    p.add_argument('--suite',action='store_true')
    args=p.parse_args()
    if args.seconds<0.04 or not __import__('math').isfinite(args.seconds):
        p.error('Require finite --seconds >= 0.04')
    model,config,path=load_run(args.model)
    out=HERE/'evaluations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
    report={'model':str(path),'model_sha256':digest(path),'environment':config,
            'platform':platform.platform(),
            'versions':{n:importlib.metadata.version(n) for n in ('mujoco','stable-baselines3','torch','gymnasium','numpy')},
            'notes':['NoSlip remains disabled; original model physics.',
                     'Deterministic predictions; nominal exact-home reset.',
                     'Velocity perturbations are state changes, not calibrated physical force pulses.',
                     'Foot distances use body origins; motion sampled at 50 Hz.',
                     'First action transition excluded from target-change RMS.'], 'trials':[]}
    cases=[('nominal',args.seconds,42,0.,None)]
    if args.suite:
        cases += [(f'velocity_seed{s}',60.,s,.02,None) for s in range(100,105)]
        cases += [(f'velocity_increment_{axis}_{sign}',60.,42,0.,(axis,sign*.05)) for axis in (0,1) for sign in (-1,1)]
    fields=['time_s','root_displacement_m','foot_separation_m','root_height_m','upright',
            'target_delta_mean_square_rad2','leg_velocity_mean_square_rad2_s2','root_angular_speed_square_rad2_s2']
    for name,seconds,seed,noise,push in cases:
        result,trajectory=evaluate(model,config,seconds,seed,noise,push)
        result['case']=name;report['trials'].append(result)
        with (out/f'{name}.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(fields);w.writerows(trajectory)
        (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result),flush=True)
    print(f'Results: {out}')


if __name__=='__main__':main()
