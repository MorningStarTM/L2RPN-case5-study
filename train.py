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
env_name = "rte_case14_realistic"  # or any other name.
env = grid2op.make(env_name, action_class=TopologyChangeAction, param=p)


agent = PPOAgent(n_actions=347, input_dims=env.observation_space.n)
trainer = PPOTrainer(agent=agent, env=env, N=20, n_epochs=50, n_games=100)

trainer.train("result\\ppo_case14.png")
#agentt.save_model("models")