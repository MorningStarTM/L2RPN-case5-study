from agents import ACAgent, KANACAgent, PPOAgent, KANPPOAgent, DQNAgent
from agents import Trainer, PPOTrainer, QTrainer
import gym


from lightsim2grid import LightSimBackend
from grid2op.Action import TopologyChangeAction
import grid2op
from grid2op.Parameters import Parameters

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

p = Parameters()
env_name = "l2rpn_case14_sandbox"  # or any other name.
env = grid2op.make(env_name+"_train", action_class=TopologyChangeAction, param=p)


agent = PPOAgent(n_actions=347, input_dims=env.observation_space.n)
trainer = PPOTrainer(agent=agent, env=env, N=70, n_epochs=200, n_games=500)

trainer.train("result\\ppo_case14.png")
#agentt.save_model("models")