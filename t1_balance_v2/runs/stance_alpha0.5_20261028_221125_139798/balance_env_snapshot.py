"""Longer-horizon balance experiment. Original MuJoCo assets/physics unchanged."""
import mujoco
import numpy as np
from gymnasium import spaces
from base_env import T1BalanceEnv


class BalanceEnv(T1BalanceEnv):
    def __init__(self, episode_seconds=30.0, alpha=1.0, reset_noise=0.005,
                 stance_weight=0.5, drift_weight=0.25, rate_weight=0.2,
                 joint_velocity_weight=0.5):
        self.origin_xy = np.zeros(2)
        self.home_width = 0.2125
        self.shadow = None
        super().__init__(episode_seconds=episode_seconds)
        self.alpha = float(alpha)
        self.reset_noise = float(reset_noise)
        self.stance_weight = float(stance_weight)
        self.drift_weight = float(drift_weight)
        self.rate_weight = float(rate_weight)
        self.joint_velocity_weight = float(joint_velocity_weight)
        if not 0 < self.alpha <= 1 or self.reset_noise < 0:
            raise ValueError("Require 0 < alpha <= 1 and nonnegative reset noise")
        if min(self.stance_weight,self.drift_weight,self.rate_weight,self.joint_velocity_weight) < 0:
            raise ValueError("Reward weights must be nonnegative")
        self.shadow = mujoco.MjData(self.model)
        self.feet = [self.model.body(n).id for n in ('left_foot_link','right_foot_link')]
        self.leg_dofs = self.model.jnt_dofadr[self.model.actuator_trnid[self.lower_body_ids,0]]
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(72,),dtype=np.float32)

    def width(self):
        if self.shadow is None:
            return self.home_width
        self.shadow.qpos[:] = self.data.qpos
        mujoco.mj_kinematics(self.model,self.shadow)
        return float(np.linalg.norm((self.shadow.xpos[self.feet[0]]-self.shadow.xpos[self.feet[1]])[:2]))

    def _observation(self):
        return np.concatenate((super()._observation(),
                               10.0*(self.data.qpos[:2]-self.origin_xy),
                               self.previous_action,
                               [10.0*(self.width()-self.home_width)])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        _,info = super().reset(seed=seed,options=options)
        self.origin_xy = self.data.qpos[:2].copy()
        self.home_width = self.width()
        if self.reset_noise:
            self.data.qvel[self.leg_dofs] += self.np_random.uniform(-self.reset_noise,self.reset_noise,len(self.leg_dofs))
            mujoco.mj_forward(self.model,self.data)
        return self._observation(),info

    def step(self, action):
        previous = self.previous_action.copy()
        raw = np.clip(np.asarray(action,dtype=np.float32),-1.0,1.0)
        applied = self.alpha*raw + (1.0-self.alpha)*previous
        obs,base_reward,terminated,truncated,info = super().step(applied)
        width = self.width()
        drift = float(np.linalg.norm(self.data.qpos[:2]-self.origin_xy))
        costs = {
            'stance': self.stance_weight*min(((width-self.home_width)/0.10)**2,9.0),
            'drift': self.drift_weight*min((drift/0.10)**2,9.0),
            'action_change_extra': (self.rate_weight-0.005)*float(np.mean((applied-previous)**2)),
            'joint_velocity': self.joint_velocity_weight*float(np.mean(self.data.qvel[self.leg_dofs]**2)),
            'root_angular_extra': (0.30-0.05)*float(np.sum(self.data.qvel[3:6]**2)),
        }
        reward = float(base_reward)-sum(costs.values())
        info.update(original_reward=float(base_reward), reward_costs=costs,
                    foot_width_m=width, root_displacement_m=drift)
        return obs,reward,terminated,truncated,info
