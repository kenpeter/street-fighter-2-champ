import retro
import pygame
import numpy as np
import sys
import time
import os

# Constants
STATE_SAVE_DIR = (
    "./StreetFighterIISpecialChampionEdition-Genesis"  # Match the path in Lobby.py
)
GAME_NAME = "StreetFighterIISpecialChampionEdition-Genesis"

# Initialize Pygame
pygame.init()

# Define window size and scaling factor
SCALE_FACTOR = 2  # Adjust for larger UI
ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 320, 224  # Genesis resolution
WINDOW_WIDTH = ORIGINAL_WIDTH * SCALE_FACTOR
WINDOW_HEIGHT = ORIGINAL_HEIGHT * SCALE_FACTOR

# Create directories if they don't exist
os.makedirs(STATE_SAVE_DIR, exist_ok=True)

# Create Pygame window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Street Fighter II - Direct Health Lock")

# Use a specific state or None for title screen
selected_state = None

# Create the Retro environment
env = retro.make(
    game=GAME_NAME,
    state=selected_state,
    use_restricted_actions=retro.Actions.ALL,
    render_mode="rgb_array",
)

# Define key mappings based on actual button order
KEY_MAP = {
    pygame.K_UP: env.buttons.index("UP") if "UP" in env.buttons else None,
    pygame.K_DOWN: env.buttons.index("DOWN") if "DOWN" in env.buttons else None,
    pygame.K_LEFT: env.buttons.index("LEFT") if "LEFT" in env.buttons else None,
    pygame.K_RIGHT: env.buttons.index("RIGHT") if "RIGHT" in env.buttons else None,
    pygame.K_z: env.buttons.index("A") if "A" in env.buttons else None,  # light punch
    pygame.K_x: env.buttons.index("B") if "B" in env.buttons else None,  # light kick
    pygame.K_a: env.buttons.index("X") if "X" in env.buttons else None,  # medium kick
    pygame.K_s: env.buttons.index("Y") if "Y" in env.buttons else None,  # heavy punch
    pygame.K_RETURN: env.buttons.index("START") if "START" in env.buttons else None,
    pygame.K_SPACE: env.buttons.index("SELECT") if "SELECT" in env.buttons else None,
    pygame.K_c: (
        env.buttons.index("C") if "C" in env.buttons else None
    ),  # additional button
    pygame.K_d: (
        env.buttons.index("Z") if "Z" in env.buttons else None
    ),  # additional button
    pygame.K_TAB: env.buttons.index("MODE") if "MODE" in env.buttons else None,
}

# Clean up None values from the KEY_MAP
KEY_MAP = {k: v for k, v in KEY_MAP.items() if v is not None}

# Reset the environment
obs = env.reset()

# Track previous key states to detect key presses
previous_keys = {}
for key in KEY_MAP.keys():
    previous_keys[key] = False

# Additional function keys
previous_keys[pygame.K_h] = False  # Toggle health lock
previous_keys[pygame.K_o] = False  # Save state
previous_keys[pygame.K_p] = False  # Load state

# For state saving
character_keys = {
    pygame.K_1: "ken.state",
    pygame.K_2: "ryu.state",
    pygame.K_3: "blanka.state",
    pygame.K_4: "chunli.state",
    pygame.K_5: "ehonda.state",
    pygame.K_6: "guile.state",
    pygame.K_7: "dhalsim.state",
    pygame.K_8: "zangief.state",
    pygame.K_9: "balrog.state",
    pygame.K_0: "vega.state",
    pygame.K_MINUS: "sagat.state",
    pygame.K_EQUALS: "bison.state",
}

# Initialize key states for all character keys
for key in character_keys.keys():
    previous_keys[key] = False

# Health lock variables
health_lock_enabled = False
MAX_HEALTH = 176  # Maximum health value in SF2

# Print instructions
print("\nDirect Health Lock Controls:")
print("- Arrow keys/ZXAS: Regular game controls")
print("- Enter: START button")
print("- H: Toggle health lock (sets health directly using the game API)")
print("- O: Save state")
print("- P: Load state")
print("- 1-9,0,-,=: Save character states")
print("- ESC: Quit")
print("\nHow to use:")
print("1. Start the game and begin a match")
print("2. Press H to enable health lock (keeps your health at maximum)")
print("3. Play through the game to reach harder opponents")
print("4. Save the state when you find a challenging opponent")
print("\nThis version uses the game's built-in health tracking instead of")
print("trying to find memory addresses. It should work more reliably.")

# Temporary state path
temp_state_path = os.path.join(STATE_SAVE_DIR, "temp.state")

# Frame counter
frame_counter = 0

# Main game loop
clock = pygame.time.Clock()
running = True

last_time = time.time()

while running:
    frame_counter += 1
    current_time = time.time()
    delta_time = current_time - last_time
    last_time = current_time

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Get current key states
    current_keys = pygame.key.get_pressed()

    # Create action array
    action = [False] * len(env.buttons)

    # Set actions based on key mappings
    for key, button_idx in KEY_MAP.items():
        if current_keys[key]:
            action[button_idx] = True

    # Handle save and load state
    if current_keys[pygame.K_o] and not previous_keys[pygame.K_o]:
        try:
            state_data = env.em.get_state()
            with open(temp_state_path, "wb") as f:
                f.write(state_data)
            print(f"Game state saved to {temp_state_path}")
        except Exception as e:
            print(f"Error saving state: {e}")

    if current_keys[pygame.K_p] and not previous_keys[pygame.K_p]:
        if os.path.exists(temp_state_path):
            try:
                with open(temp_state_path, "rb") as f:
                    state_data = f.read()
                env.em.set_state(state_data)
                print(f"Game state loaded from {temp_state_path}")
            except Exception as e:
                print(f"Error loading state: {e}")
        else:
            print(f"No saved state found at {temp_state_path}")

    # Handle character state saving
    for key, state_name in character_keys.items():
        if current_keys[key] and not previous_keys[key]:
            try:
                state_path = os.path.join(STATE_SAVE_DIR, state_name)
                state_data = env.em.get_state()
                with open(state_path, "wb") as f:
                    f.write(state_data)
                print(f"Game state saved as {state_path} (compatible with Lobby.py)")
            except Exception as e:
                print(f"Error saving state as {state_name}: {e}")
            finally:
                previous_keys[key] = current_keys[key]

    # Toggle health lock
    if current_keys[pygame.K_h] and not previous_keys[pygame.K_h]:
        health_lock_enabled = not health_lock_enabled
        print(f"Health lock {'enabled' if health_lock_enabled else 'disabled'}")

    # Update previous keys
    for key in previous_keys:
        previous_keys[key] = current_keys[key]

    # Step the environment
    try:
        step_result = env.step(action)
        if len(step_result) == 4:  # older pattern
            obs, reward, done, info = step_result
            terminated = done
        else:  # newer pattern with 5 returns
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        # Apply health lock by directly setting player health in next step
        if health_lock_enabled and "health" in info and info["health"] < MAX_HEALTH:
            # We can't modify 'info' directly as it's read-only
            # But we can use env.data.set_value to set it for the next frame
            try:
                # First method - try to directly set the value via environment data
                env.data.set_value("health", MAX_HEALTH)
                if frame_counter % 300 == 0:  # Log every 5 seconds
                    print(f"Set health to {MAX_HEALTH} via data API")
            except Exception as e:
                if frame_counter % 300 == 0:  # Log every 5 seconds
                    print(f"Error setting health: {e}")

                # Second method - use the game's own health tracker to track where it gets health from
                try:
                    # Keep track of the game's own health data
                    current_health = info.get("health", 0)
                    if current_health < MAX_HEALTH and current_health >= 0:
                        print(
                            f"Health decreased to {current_health}, trying to restore..."
                        )
                        # Try alternative approaches that might work
                        try:
                            # Method 1: Set info data for next frame
                            data = env.data
                            if hasattr(data, "_Retro__info"):
                                data._Retro__info["health"] = MAX_HEALTH
                                print("Set health via __info")
                        except:
                            pass

                        try:
                            # Method 2: Use data.set_value again
                            env.data.set_value("health", MAX_HEALTH)
                            print("Set health via set_value")
                        except:
                            pass
                except Exception as e:
                    pass
    except Exception as e:
        print(f"Error in env.step: {e}")
        break

    # Render the game
    try:
        frame = env.render()
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(
            frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
    except Exception as e:
        print(f"Error in rendering: {e}")
        break

    # Print game info occasionally
    if frame_counter % 300 == 0:  # Every ~5 seconds at 60fps
        if "info" in locals() and info:
            print(f"Game info: {info}")

            # If health lock is enabled, print status
            if health_lock_enabled:
                print(
                    f"Health lock active: Current health = {info.get('health', 'unknown')}"
                )

                # If health is not max, try to set it directly again
                if (
                    "health" in info
                    and info["health"] < MAX_HEALTH
                    and info["health"] >= 0
                ):
                    try:
                        env.data.set_value("health", MAX_HEALTH)
                        print(f"Restored health to {MAX_HEALTH}")
                    except Exception as e:
                        print(f"Failed to restore health: {e}")

    # Cap the framerate
    clock.tick(60)

    # Break the loop if the game is done
    if "done" in locals() and done:
        print("Game over! Resetting...")
        obs = env.reset()

# Close the environment and Pygame
env.close()
pygame.quit()
sys.exit(0)
