#!/usr/bin/env python3
import retro
import os
import sys
import time
import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description="Create state files for stable-retro that work with Lobby.py"
    )
    parser.add_argument(
        "--character", type=str, default="ken", help="Character name for the state file"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test if the state loads correctly"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available built-in states"
    )
    return parser


# Constants
GAME_NAME = "StreetFighterIISpecialChampionEdition-Genesis"
STATE_DIR = os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis")

# Create state directory if it doesn't exist
os.makedirs(STATE_DIR, exist_ok=True)
print(f"State directory: {STATE_DIR}")


def list_available_states():
    """List built-in states available in retro"""
    try:
        states = retro.data.list_states(GAME_NAME)
        print(f"Built-in states for {GAME_NAME}:")
        for state in states:
            print(f"- {state}")
        return states
    except Exception as e:
        print(f"Error listing states: {e}")
        return []


def create_state_file(character_name):
    """Create a state file using retro's API"""
    # Create full path for the state file
    state_file_path = os.path.join(STATE_DIR, f"{character_name}.state")
    print(f"Creating state file: {state_file_path}")

    try:
        # Make the game environment
        env = retro.make(game=GAME_NAME)
        print("Created environment")

        # Reset the environment
        obs = env.reset()
        print("Reset environment")

        # Take a few no-op steps to ensure the environment is stable
        print("Taking a few steps to stabilize the environment...")
        for _ in range(10):
            env.step([0] * len(env.buttons))

        # Get the current state
        state_data = env.em.get_state()
        print(f"Got state data of length: {len(state_data)}")

        # Save the state
        with open(state_file_path, "wb") as f:
            f.write(state_data)

        print(f"State file saved to: {state_file_path}")
        env.close()

        # Also save a copy with just the name (no extension) to ensure compatibility
        # with various ways the state might be loaded
        state_name_path = os.path.join(STATE_DIR, character_name)
        with open(state_name_path, "wb") as f:
            f.write(state_data)
        print(f"Also saved state as: {state_name_path}")

        return state_file_path

    except Exception as e:
        print(f"Error creating state file: {e}")
        return None


def test_state_directly():
    """Test if a state loads directly using retro.make"""
    try:
        # Try the state with a .state extension first
        state_path = os.path.join(STATE_DIR, "ken.state")
        print(f"Testing state with path: {state_path}")

        env = retro.make(game=GAME_NAME, state=state_path)
        obs = env.reset()
        print("State loaded successfully with full path!")
        env.close()

        # Try with just the state name
        print(f"Testing state with name only: ken")
        env = retro.make(game=GAME_NAME, state="ken")
        obs = env.reset()
        print("State loaded successfully with name only!")
        env.close()

        # Try with the state name path
        state_name_path = os.path.join(STATE_DIR, "ken")
        print(f"Testing state with name path: {state_name_path}")
        env = retro.make(game=GAME_NAME, state=state_name_path)
        obs = env.reset()
        print("State loaded successfully with name path!")
        env.close()

        return True

    except Exception as e:
        print(f"Error testing state: {e}")
        return False


def create_default_state():
    """Create a default state file named 'default' for fallback"""
    state_file_path = os.path.join(STATE_DIR, "default.state")
    print(f"Creating default state file: {state_file_path}")

    try:
        # Make the game environment
        env = retro.make(game=GAME_NAME)
        print("Created environment")

        # Reset the environment
        obs = env.reset()
        print("Reset environment")

        # Get the current state
        state_data = env.em.get_state()

        # Save the state
        with open(state_file_path, "wb") as f:
            f.write(state_data)

        # Also save without extension
        with open(os.path.join(STATE_DIR, "default"), "wb") as f:
            f.write(state_data)

        print(f"Default state file saved")
        env.close()
        return state_file_path

    except Exception as e:
        print(f"Error creating default state file: {e}")
        return None


def register_state_dir():
    """Register the state directory with retro to ensure states are found"""
    try:
        import retro.data

        retro.data.add_custom_path(STATE_DIR)
        print(f"Registered state directory: {STATE_DIR}")
        return True
    except Exception as e:
        print(f"Error registering state directory: {e}")
        return False


def patch_retro_make():
    """Patch retro.make to better handle state paths"""
    try:
        # Only do this if necessary
        original_make = retro.make

        def patched_make(game, state=None, **kwargs):
            if state and not os.path.exists(state) and not state.startswith("/"):
                # Try with .state extension
                state_with_ext = os.path.join(STATE_DIR, f"{state}.state")
                if os.path.exists(state_with_ext):
                    print(f"Using state file: {state_with_ext}")
                    return original_make(game, state=state_with_ext, **kwargs)

                # Try without extension
                state_no_ext = os.path.join(STATE_DIR, state)
                if os.path.exists(state_no_ext):
                    print(f"Using state file: {state_no_ext}")
                    return original_make(game, state=state_no_ext, **kwargs)

            return original_make(game, state=state, **kwargs)

        retro.make = patched_make
        print("Patched retro.make to better handle state paths")
        return True
    except Exception as e:
        print(f"Error patching retro.make: {e}")
        return False


if __name__ == "__main__":
    args = create_parser().parse_args()

    print("=" * 50)
    print("State Creator for Street Fighter II")
    print("=" * 50)

    # Register the state directory
    register_state_dir()

    # Patch retro.make
    patch_retro_make()

    if args.list:
        list_available_states()
        sys.exit(0)

    if args.test:
        if test_state_directly():
            print(
                "\n✅ State testing successful! Your state files should work with Lobby.py"
            )
        else:
            print("\n❌ State testing failed. Let's try creating new state files...")
            create_state_file(args.character)
            test_state_directly()
    else:
        # Create a state file for the specified character
        create_state_file(args.character)

        # Always create a default state as a fallback
        create_default_state()

    print("\nDone! If you're using Lobby.py, try running:")
    print("python DeepQAgent.py")
