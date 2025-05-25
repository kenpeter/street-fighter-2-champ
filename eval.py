#!/usr/bin/env python3
"""
Simple script to watch your agent play 3 games
Just load model, load state, play games with visual UI
"""

import retro
import numpy as np
import time
import os
import random
from DeepQAgent import DeepQAgent, Moves
from enum import Enum

# Set random seed for different behavior each game
random.seed(time.time())
np.random.seed(int(time.time()) % 2**32)


# 12-action moves (matching your trained model)
class Moves12(Enum):
    Idle = 0
    Right = 1
    Left = 2
    Down = 3
    LightPunch = 4
    MediumPunch = 5
    HeavyPunch = 6
    LightKick = 7
    MediumKick = 8
    HeavyKick = 9
    CrouchLightPunch = 10
    CrouchMediumPunch = 11


Moves12Dict = {
    Moves12.Idle: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    Moves12.Right: [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    Moves12.Left: [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    Moves12.Down: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
    Moves12.LightPunch: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
    Moves12.MediumPunch: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
    Moves12.HeavyPunch: [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    Moves12.LightKick: [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
    Moves12.MediumKick: [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    Moves12.HeavyKick: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    Moves12.CrouchLightPunch: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]],
    Moves12.CrouchMediumPunch: [[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]],
}


class SimpleAgent:
    def __init__(self, model_path):
        # Create base agent
        self.agent = DeepQAgent(stateSize=60, total_timesteps=1000)
        self.agent.actionSize = 12  # Fix action size

        # Load model
        self.agent.model.load_weights(model_path)
        print(f"✅ Loaded model: {model_path}")

    def get_action(self, info):
        # Add some randomness for variety between games
        if random.random() < 0.1:  # 10% chance for random action
            action_index = random.randint(0, 11)
            move = list(Moves12)[action_index]
            frame_inputs = Moves12Dict[move]
            return action_index, frame_inputs, move.name + " (random)"

        # Get Q-values
        state_data = self.agent.prepareNetworkInputs(info)
        q_values = self.agent.model.predict(state_data, verbose=0)[0]

        # Add some noise to Q-values for variety
        noise = np.random.normal(0, 0.1, q_values.shape)
        q_values_noisy = q_values + noise

        # Pick best action from noisy Q-values
        action_index = np.argmax(q_values_noisy)
        move = list(Moves12)[action_index]
        frame_inputs = Moves12Dict[move]

        return action_index, frame_inputs, move.name


def play_game(agent, state_name, game_num):
    print(f"\n🎮 GAME {game_num}: Using state '{state_name}'")

    # Set different random seed for each game
    game_seed = int(time.time() * 1000) + game_num
    random.seed(game_seed)
    np.random.seed(game_seed % 2**32)
    print(f"🎲 Game seed: {game_seed}")

    # Create environment with rendering
    env = retro.make("StreetFighterIISpecialChampionEdition-Genesis", players=1)
    env.reset()

    # Load state
    state_path = f"./StreetFighterIISpecialChampionEdition-Genesis/{state_name}.state"
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state_data = f.read()
        env.em.set_state(state_data)
        print(f"✅ Loaded state: {state_name}")
    else:
        print(f"⚠️  State not found, using default")

    # Initialize with some random steps to create variety
    print("🔄 Randomizing start position...")
    for _ in range(random.randint(5, 25)):  # 5-25 random steps
        random_action = [0] * 12
        if random.random() < 0.3:  # 30% chance to do something
            button_idx = random.randint(0, 11)
            random_action[button_idx] = 1
        env.step(random_action)
        env.render()
        time.sleep(0.01)

    # Initialize
    step_count = 0
    max_steps = 1000
    done = False

    while not done and step_count < max_steps:
        # Render game
        env.render()
        time.sleep(0.016)  # ~60 FPS

        # Get current info
        step_result = env.step([0] * 12)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        # Get agent action
        action_index, frame_inputs, move_name = agent.get_action(info)

        # Show what agent is doing (every 60 steps)
        if step_count % 60 == 0:
            player_health = info.get("health", 0)
            enemy_health = info.get("enemy_health", 0)
            print(
                f"Step {step_count}: {move_name} | Player: {player_health}, Enemy: {enemy_health}"
            )

        # Execute action (hold for multiple frames)
        for frame in frame_inputs:
            for _ in range(8):  # Hold button for 8 frames
                step_result = env.step(frame)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated

                env.render()
                time.sleep(0.016)

                if (
                    done
                    or info.get("health", 0) <= 0
                    or info.get("enemy_health", 0) <= 0
                ):
                    done = True
                    break
            if done:
                break

        step_count += 1

    # Game result
    final_player_health = info.get("health", 0)
    final_enemy_health = info.get("enemy_health", 0)

    if final_player_health > final_enemy_health:
        result = "WIN"
    elif final_enemy_health > final_player_health:
        result = "LOSS"
    else:
        result = "DRAW"

    print(f"🏁 Game {game_num} Result: {result}")
    print(f"   Final - Player: {final_player_health}, Enemy: {final_enemy_health}")
    print(f"   Steps: {step_count}")

    env.close()
    return result


def main():
    print("🎮 SIMPLE AGENT VIEWER - WATCH 3 GAMES")
    print("=" * 50)

    # Load agent
    model_path = "models/DeepQAgentModel_200000.weights.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    agent = SimpleAgent(model_path)

    # Get available states
    state_dir = "./StreetFighterIISpecialChampionEdition-Genesis"
    if os.path.exists(state_dir):
        states = [f[:-6] for f in os.listdir(state_dir) if f.endswith(".state")]
        print(f"Available states: {states}")
    else:
        states = ["default"]

    # Play 3 games
    results = []
    for game_num in range(1, 4):
        state_name = random.choice(states)
        result = play_game(agent, state_name, game_num)
        results.append(result)

        # Short pause between games
        print("Press Enter for next game...")
        input()

    # Final results
    wins = results.count("WIN")
    losses = results.count("LOSS")
    draws = results.count("DRAW")

    print(f"\n🏆 FINAL RESULTS:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win Rate: {wins/3*100:.1f}%")


if __name__ == "__main__":
    main()
