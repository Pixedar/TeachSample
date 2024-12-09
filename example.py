import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# #### TRAINING ####
# # # Instantiate the agent
# model = DQN("MlpPolicy", env, verbose=1)
# # Train the agent and display a progress bar
# model.learn(total_timesteps=int(2e5), progress_bar=True)
# # Save the agent
# model.save("dqn_lunar_clear")
# del model  # delete trained model to demonstrate loading
# #### TRAINING ####


# Load the trained agent
#Not trained madel
# model = DQN.load("dqn_lunar_clear", env=env)
#Trained model
model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")