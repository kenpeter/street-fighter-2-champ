import retro
import pygame
import numpy as np
import sys
import time

# Initialize Pygame
pygame.init()

# Define window size and scaling factor
SCALE_FACTOR = 2  # Adjust for larger UI
ORIGINAL_WIDTH, ORIGINAL_HEIGHT = 320, 224  # Genesis resolution
WINDOW_WIDTH = ORIGINAL_WIDTH * SCALE_FACTOR
WINDOW_HEIGHT = ORIGINAL_HEIGHT * SCALE_FACTOR

# Create Pygame window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Street Fighter II")

# List available states
game_name = "StreetFighterIISpecialChampionEdition-Genesis"
available_states = retro.data.list_states(game_name)
print(f"Available states for {game_name}:")
for state in available_states:
    print(f"- {state}")

# Use a specific state
# Common states are often named like 'CharacterSelect', 'Level1', etc.
# If none of the available states give you the character select,
# use the default state (just pass None or empty string)
selected_state = None  # Try using None first, this often starts at the title screen

# Create the Retro environment
env = retro.make(
    game=game_name,
    state=selected_state,  # Use the selected state or default
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
print("- ESC: Quit game")

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
