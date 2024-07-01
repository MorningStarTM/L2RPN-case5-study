from agents import ACAgent, KANACAgent, PPOAgent, KANPPOAgent, DQNAgent
from agents import Trainer, PPOTrainer, QTrainer
import gym


from lightsim2grid import LightSimBackend
from grid2op.Action import TopologyChangeAction
import grid2op
from grid2op.Parameters import Parameters
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

p = Parameters()
env_name = "rte_case5_example"  # or any other name.
#env = grid2op.make(env_name+"_train", action_class=TopologyChangeAction, param=p)
env = grid2op.make(env_name, test=True, action_class=TopologyChangeAction, param=p)


agent = DQNAgent()
trainer = QTrainer(agent=agent, env=env, n_episode=100)
#trainer = PPOTrainer(agent=agent, env=env, N=70, n_epochs=200, n_games=300)

trainer.train("result\\dqn_case5.png", model_path="dqn")
#agentt.save_model("models")