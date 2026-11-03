"""Train/evaluate isolated longer-horizon PPO candidates with original physics."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from balance_env import BalanceEnv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = {
    'quiet': 'experiments/smooth_balance_20261027_221853_406565/control/checkpoint_000050176.zip',
    'stance': 'experiments/stance_balance_20261027_224155_950849/stance/checkpoint_000075264.zip',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def preserved_state(model):
    py, npstate, ts, mode = random.getstate(),np.random.get_state(),torch.random.get_rng_state(),model.policy.training
    try:
        yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.random.set_rng_state(ts);model.policy.set_training_mode(mode)


def evaluate(model,config,seconds=60.0,seed=42,noise=0.0,push=None):
    env=BalanceEnv(episode_seconds=seconds,reset_noise=noise,**config)
    rows=[];previous=None;applied_push=False
    try:
        with preserved_state(model):
            obs,_=env.reset(seed=seed)
            for i in range(env.max_episode_steps):
                if push is not None and not applied_push and i*0.02>=5.0:
                    env.data.qvel[push[0]]+=push[1]
                    import mujoco
                    mujoco.mj_forward(env.model,env.data)
                    obs=env._observation();applied_push=True
                action,_=model.predict(obs,deterministic=True)
                obs,_,term,trunc,info=env.step(action)
                targets=env.data.ctrl[env.lower_body_ids].copy()
                rows.append([float(env.data.time),info['root_displacement_m'],info['foot_width_m'],
                             info['root_height'],info['upright_score'],
                             np.nan if previous is None else np.mean((targets-previous)**2),
                             np.mean(env.data.qvel[env.leg_dofs]**2),np.sum(env.data.qvel[3:6]**2)])
                previous=targets
                if term or trunc:break
        x=np.asarray(rows);late=x[x[:,0]>5];last=x[x[:,0]>=max(0,x[-1,0]-10)]
        result=dict(duration_s=float(x[-1,0]),fell=bool(term),max_drift_m=float(x[:,1].max()),
                    final_width_m=float(x[-1,2]),max_width_error_m=float(np.max(abs(x[:,2]-env.home_width))),
                    minimum_height_m=float(x[:,3].min()),minimum_upright=float(x[:,4].min()),
                    target_delta_rms_rad=float(np.sqrt(np.nanmean(x[:,5]))) if len(x)>1 else None,
                    leg_velocity_rms_rad_s=float(np.sqrt(x[:,6].mean())),
                    root_angular_speed_rms_rad_s=float(np.sqrt(x[:,7].mean())),
                    after5_leg_velocity_rms_rad_s=float(np.sqrt(late[:,6].mean())) if len(late) else None,
                    last10_width_slope_m_s=float(np.polyfit(last[:,0],last[:,2],1)[0]) if len(last)>1 else None,
                    noslip_iterations=int(env.model.opt.noslip_iterations),seed=seed,reset_noise=noise,push=push)
        return result,x
    finally:
        env.close()


def initialize(source,env,seed,std):
    old=PPO.load(str(source),device='cpu')
    model=PPO('MlpPolicy',env,seed=seed,device='cpu',learning_rate=1e-4,
              n_steps=1024,batch_size=256,n_epochs=5,gamma=0.995,gae_lambda=0.95,
              ent_coef=0.0,clip_range=0.1,target_kl=0.02,
              policy_kwargs={'net_arch':[128,128]},verbose=0)
    state=model.policy.state_dict();old_state=old.policy.state_dict()
    for name,value in state.items():
        source_value=old_state[name]
        if value.shape==source_value.shape:
            value.copy_(source_value)
        elif value.ndim==2 and value.shape[0]==source_value.shape[0] and value.shape[1]==72 and source_value.shape[1]==57:
            value.zero_();value[:,:57].copy_(source_value)
        else:
            raise ValueError(f'Unexpected policy shape: {name}')
    model.policy.load_state_dict(state)
    with torch.no_grad():
        model.policy.log_std.fill_(np.log(std))
    # The added features initially have zero weight: deterministic means must match.
    rng=np.random.default_rng(123)
    probe=rng.normal(size=(32,57)).astype(np.float32)
    extended=np.concatenate((probe,np.zeros((32,15),np.float32)),axis=1)
    np.testing.assert_allclose(old.predict(probe,deterministic=True)[0],
                               model.predict(extended,deterministic=True)[0],atol=2e-6,rtol=2e-6)
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=SOURCES,default='quiet')
    p.add_argument('--alpha',type=float,default=1.0)
    p.add_argument('--steps',type=int,default=200000)
    p.add_argument('--eval-every',type=int,default=25000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--std',type=float,default=0.35)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.steps<1 or args.eval_every<1 or not 0<args.std<=2:
        p.error('Require positive steps/eval-every and 0 < std <= 2')
    torch.set_num_threads(1)
    config=dict(alpha=args.alpha,stance_weight=0.5,drift_weight=0.25,rate_weight=0.2,joint_velocity_weight=0.5)
    raw=BalanceEnv(**config);check_env(raw,warn=True)
    source=ROOT/SOURCES[args.source]
    model=initialize(source,raw,args.seed,args.std)
    if args.check_only:
        print('Environment check and 57-to-72 observation policy-transfer check passed. No training.');raw.close();return
    out=HERE/'runs'/f'{args.source}_alpha{args.alpha:g}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True,exist_ok=False)
    env=Monitor(raw,filename=str(out/'training'));model.set_env(env)
    model.set_logger(configure(str(out),['csv']))
    files=[source,HERE/'base_env.py',HERE/'balance_env.py']+list((ROOT/'assets/mujoco_menagerie/booster_t1').rglob('*.xml'))
    hashes={str(f.relative_to(ROOT)):digest(f) for f in files}
    report={'settings':vars(args),'environment':config,'training_episode_seconds':30.0,
            'training_reset_velocity_noise_rad_s':0.005,'input_sha256':hashes,
            'versions':{n:importlib.metadata.version(n) for n in ('mujoco','stable-baselines3','torch','gymnasium','numpy')},
            'notes':['Original XML/solver/friction/gains/limits unchanged.',
                     '72 observations: original 57 + scaled root XY error + previous applied actions + scaled foot-width error.',
                     'Policy/value weights transferred; new optimizer and exploration std; added input weights start at zero.',
                     'Alpha<1 changes the action interface during both training and evaluation.',
                     'Nominal checkpoint screening is not a robustness result.'], 'evaluations':[]}
    t=time.monotonic()
    try:
        targets=[0]+list(range(args.eval_every,args.steps,args.eval_every))+[args.steps]
        for target in targets:
            if target>model.num_timesteps:
                model.learn(total_timesteps=target-model.num_timesteps,reset_num_timesteps=False)
            result,trajectory=evaluate(model,config)
            checkpoint=out/f'checkpoint_{model.num_timesteps:09d}.zip';model.save(checkpoint)
            np.savez_compressed(out/f'trajectory_{model.num_timesteps:09d}.npz',data=trajectory)
            result.update(steps=int(model.num_timesteps),checkpoint=checkpoint.name,elapsed_s=time.monotonic()-t)
            report['evaluations'].append(result)
            (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps({'branch':out.name,**result}),flush=True)
        assert all(digest(ROOT/k)==v for k,v in hashes.items()),'An original input changed'
        report['original_inputs_unchanged']=True
        (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f'Complete: {out}',flush=True)
    finally:
        env.close()


if __name__=='__main__':main()
