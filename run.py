from rl4co.envs.routing import WCRPEnv, WCRPGenerator
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.utils import RL4COTrainer

# Call the train function directly from inside the package
# You can also pass additional Hydra arguments, like:
# `python run.py experiment=routing/am env=cvrp env.num_loc=50`
# Alternatively, you may run without Hydra (see examples/1.quickstart.ipynb)
if __name__ == "__main__":
    gen = WCRPGenerator(num_loc=20, loc_distribution="uniform", prize_distribution="uniform")
    env = WCRPEnv(gen)
    pol = AttentionModelPolicy(env_name=env.name, num_encoder_layers=3)
    model = REINFORCE(env, pol, batch_size=256, optimizer_kwargs={"lr": 1e-4})
    trainer = RL4COTrainer(max_epochs=10, accelerator="gpu")
    trainer.fit(model)