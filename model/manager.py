import gymnasium
import numpy as np
from model.agent import Agent


def run():
    env = gymnasium.make("LunarLanderContinuous-v3")
    agent = Agent(
        alpha=0.001,
        beta=0.001,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        batch_size=100,
        layer1_size=400,
        layer2_size=300,
        n_actions=env.action_space.shape[0],
    )

    n_games = 1000
    filename = "plots/" + "LunarLanderContinous_" + str(n_games) + "_games.png"

    best_score = -float("inf")  # Use negative infinity as starting point
    score_history = []

    # agent.load_models()

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_model()

        print("episode ", i, " score %.1f" % score, " average score %1.f" % avg_score)

    x = [i + 1 for i in range(n_games)]
