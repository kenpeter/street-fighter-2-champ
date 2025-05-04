"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
All credit too open-ai's examples: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
"""

# gym, numpy, retro
import gym
import numpy as np
import retro


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    # list of valid button combo
    def __init__(self, env, combos):
        # init the env
        super().__init__(env)
        #
        assert isinstance(env.action_space, gym.spaces.MultiBinary)

        # button from env
        buttons = env.unwrapped.buttons

        # useful combo
        self._decode_discrete_action = []
        # combo
        self._combos = combos

        # we have diff combos
        for combo in combos:
            # all of them false
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                # because combo, we press those button
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

    def get_action_meaning(self, act):
        return self._combos[act]


class StreetFighter2Discretizer(Discretizer):
    """
    Use Street Fighter 2
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """

    # A, B, C, punches
    # X, Y, Z, kickes
    def __init__(self, env):
        super().__init__(
            env=env,
            combos=[
                [],
                ["UP"],
                ["DOWN"],
                ["LEFT"],
                ["UP", "LEFT"],
                ["DOWN", "LEFT"],
                ["RIGHT"],
                ["UP", "RIGHT"],
                ["DOWN", "RIGHT"],
                ["B"],
                ["B", "DOWN"],
                ["B", "LEFT"],
                ["B", "RIGHT"],
                ["A"],
                ["A", "DOWN"],
                ["A", "LEFT"],
                ["A", "RIGHT"],
                ["C"],
                ["DOWN", "C"],
                ["LEFT", "C"],
                ["RIGHT", "C"],
                ["Y"],
                ["DOWN", "Y"],
                ["LEFT", "Y"],
                ["DOWN", "LEFT", "Y"],
                ["RIGHT", "Y"],
                ["X"],
                ["DOWN", "X"],
                ["LEFT", "X"],
                ["DOWN", "LEFT", "X"],
                ["RIGHT", "X"],
                ["DOWN", "RIGHT", "X"],
                ["Z"],
                ["DOWN", "Z"],
                ["LEFT", "Z"],
                ["DOWN", "LEFT", "Z"],
                ["RIGHT", "Z"],
                ["DOWN", "RIGHT", "Z"],
            ],
        )


"""
    Initializes an example discrete environment and randomly selects moves for the agent to make.
    The meaning of each selected move in terms of what buttons are being pressed is also displayed.
"""


def main():
    env = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis")
    # from above
    env = StreetFighter2Discretizer(env)
    print(env.action_space)
    print(env.action_space.sample())
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        _, _, _, info = env.step(action)
        print(info["status"])
        print(env.get_action_meaning(action))
        input()

    env.close()


if __name__ == "__main__":
    main()
