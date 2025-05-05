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
pygame.display.set_caption("Street Fighter II - State Creator")

# List available states
available_states = retro.data.list_states(GAME_NAME)
print(f"Available states for {GAME_NAME}:")
for state in available_states:
    print(f"- {state}")

# Use a specific state
# Use None for title screen or specify a state if you want to start at a specific point
selected_state = None

# Create the Retro environment
env = retro.make(
    game=GAME_NAME,
    state=selected_state,
    use_restricted_actions=retro.Actions.ALL,
    render_mode="rgb_array",
)

# Print the actual button mapping for verification
print("Button order in environment:")
print(env.buttons)

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

# Print the key mappings for reference
print("\nKey mappings:")
for key, button_idx in KEY_MAP.items():
    key_name = pygame.key.name(key)
    button_name = env.buttons[button_idx]
    print(f"{key_name} -> {button_name} (index {button_idx})")

# Reset the environment
obs = env.reset()

# Track previous key states to detect key presses
previous_keys = {}
for key in KEY_MAP.keys():
    previous_keys[key] = False

# Keys for save state functionality
previous_keys[pygame.K_o] = False  # Save state
previous_keys[pygame.K_p] = False  # Load state
previous_keys[pygame.K_1] = False  # Save as "ken.state"
previous_keys[pygame.K_2] = False  # Save as "ryu.state"
previous_keys[pygame.K_3] = False  # Save as "blanka.state"
previous_keys[pygame.K_4] = False  # Save as "chunli.state"

# Main game loop
clock = pygame.time.Clock()
running = True

print("\nControls:")
print("- Arrow keys: Movement")
print("- Z: A (light punch)")
print("- X: B (light kick)")
print("- A: X (medium kick)")
print("- S: Y (heavy punch)")
print("- Enter: START (press this to start the game from title screen)")
print("- C: C button")
print("- D: Z button")
print("- Tab: MODE button")
print("- O: Save temporary state")
print("- P: Load temporary state")
print("- 1: Save as 'ken.state' (compatible with Lobby.py)")
print("- 2: Save as 'ryu.state' (compatible with Lobby.py)")
print("- 3: Save as 'blanka.state' (compatible with Lobby.py)")
print("- 4: Save as 'chunli.state' (compatible with Lobby.py)")
print("- ESC: Quit game")

# Temporary state path
temp_state_path = os.path.join(STATE_SAVE_DIR, "temp.state")

# Frame counter for debug
frame_counter = 0

while running:
    frame_counter += 1

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Get current key states
    current_keys = pygame.key.get_pressed()

    # Create action array matching the exact size of env.buttons
    action = [False] * len(env.buttons)

    # Set actions based on key mappings
    for key, button_idx in KEY_MAP.items():
        if current_keys[key]:
            action[button_idx] = True

    # Debug key presses - using our manual previous_keys dictionary
    for key, button_idx in KEY_MAP.items():
        if current_keys[key] and not previous_keys[key]:
            print(f"Pressed: {pygame.key.name(key)} -> {env.buttons[button_idx]}")
        previous_keys[key] = current_keys[key]

    # Handle temporary save state (O key)
    if current_keys[pygame.K_o] and not previous_keys[pygame.K_o]:
        try:
            # Use the retro library's save_state method
            state_data = env.em.get_state()
            with open(temp_state_path, "wb") as f:
                f.write(state_data)
            print(f"Game state saved to {temp_state_path}")
        except Exception as e:
            print(f"Error saving state: {e}")

    # Handle temporary load state (P key)
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

    # Save as specific character states (for Lobby.py compatibility)
    character_keys = {
        pygame.K_1: "ken.state",
        pygame.K_2: "ryu.state",
        pygame.K_3: "blanka.state",
        pygame.K_4: "chunli.state",
    }

    for key, state_name in character_keys.items():
        if current_keys[key] and not previous_keys[key]:
            try:
                # Save the current state in the format expected by Lobby.py
                state_path = os.path.join(STATE_SAVE_DIR, state_name)
                state_data = env.em.get_state()
                with open(state_path, "wb") as f:
                    f.write(state_data)
                print(f"Game state saved as {state_path} (compatible with Lobby.py)")
            except Exception as e:
                print(f"Error saving state as {state_name}: {e}")
            finally:
                previous_keys[key] = current_keys[key]

    # Update O and P key states
    previous_keys[pygame.K_o] = current_keys[pygame.K_o]
    previous_keys[pygame.K_p] = current_keys[pygame.K_p]

    # Step the environment
    try:
        # Handle different return patterns (older vs newer retro versions)
        step_result = env.step(action)
        if len(step_result) == 4:  # older pattern
            obs, reward, done, info = step_result
            terminated = done
        else:  # newer pattern with 5 returns
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
    except Exception as e:
        print(f"Error in env.step: {e}")
        break

    # Render the game
    try:
        frame = env.render()

        # Convert to Pygame surface
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        # Scale and display
        scaled_surface = pygame.transform.scale(
            frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
    except Exception as e:
        print(f"Error in rendering: {e}")
        print(f"Frame shape: {frame.shape if 'frame' in locals() else 'unknown'}")
        break

    # Print game info occasionally
    if frame_counter % 300 == 0:  # Every ~5 seconds at 60fps
        print(f"Game info: {info}")

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
