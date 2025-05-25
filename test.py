import retro
import os
import time
import numpy as np


def read_ram_values(env, info):
    """Read game state from RAM"""
    ram_info = {
        "enemy_health": {"address": 16745154, "type": ">i2"},
        "enemy_x_position": {"address": 16745094, "type": ">u2"},
        "enemy_y_position": {"address": 16745098, "type": ">u2"},
        "health": {"address": 16744514, "type": ">i2"},
        "x_position": {"address": 16744454, "type": ">u2"},
        "y_position": {"address": 16744458, "type": ">u2"},
        "status": {"address": 16744450, "type": ">u2"},
    }

    try:
        if hasattr(env.unwrapped, "get_ram"):
            ram = env.unwrapped.get_ram()
        elif hasattr(env.unwrapped, "em") and hasattr(env.unwrapped.em, "get_ram"):
            ram = env.unwrapped.em.get_ram()
        else:
            return info

        for key, address_info in ram_info.items():
            addr = address_info["address"]
            data_type = address_info["type"]
            if addr >= len(ram):
                continue
            try:
                if data_type == ">u2":
                    if addr + 1 < len(ram):
                        value = (ram[addr] << 8) | ram[addr + 1]
                    else:
                        continue
                elif data_type == ">i2":
                    if addr + 1 < len(ram):
                        value = (ram[addr] << 8) | ram[addr + 1]
                        if value >= 32768:
                            value -= 65536
                    else:
                        continue
                else:
                    value = ram[addr]
                info[key] = value
            except Exception:
                pass
    except Exception:
        pass

    # Set defaults if not found
    defaults = {
        "enemy_health": 176,
        "enemy_x_position": 200,
        "enemy_y_position": 0,
        "health": 176,
        "x_position": 100,
        "y_position": 0,
        "status": 512,
    }
    for key, default_value in defaults.items():
        if key not in info:
            info[key] = default_value

    return info


def test_movement():
    """Test basic movement commands"""
    print("🧪 MOVEMENT DIAGNOSTIC TEST")
    print("=" * 50)

    # Create environment
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = retro.make(game=game, players=1)
    env.reset()

    # Load state if available
    state_path = os.path.join(
        os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis"),
        "ken_bison_12.state",
    )
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state_data = f.read()
        env.em.set_state(state_data)
        print("✅ Loaded state file")
    else:
        print("⚠️  No state file found, using default")

    # Button mappings
    # ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    #  [0,   1,   2,     3,      4,    5,     6,      7,       8,   9,   10,  11]

    test_inputs = {
        "IDLE": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "LEFT": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "RIGHT": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "UP": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "DOWN": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "PUNCH": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # X button
        "KICK": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # C button
    }

    # Initialize and get starting position
    print("\n📍 INITIALIZING...")
    for _ in range(30):  # Let game settle
        step_result = env.step(test_inputs["IDLE"])
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        if hasattr(env, "render"):
            env.render()

    info = read_ram_values(env, info)
    start_x = info.get("x_position", 0)
    start_y = info.get("y_position", 0)
    start_health = info.get("health", 176)

    print(f"🎮 STARTING POSITION:")
    print(f"   Player: ({start_x}, {start_y})")
    print(
        f"   Enemy: ({info.get('enemy_x_position', 0)}, {info.get('enemy_y_position', 0)})"
    )
    print(f"   Health: {start_health}")
    print(f"   Status: {info.get('status', 0)}")

    # Test each input
    for input_name, input_array in test_inputs.items():
        if input_name == "IDLE":
            continue

        print(f"\n🧪 TESTING {input_name}: {input_array}")

        # Record position before
        before_info = read_ram_values(env, {})
        before_x = before_info.get("x_position", start_x)
        before_y = before_info.get("y_position", start_y)
        before_health = before_info.get("health", start_health)

        print(f"   Before: Pos=({before_x}, {before_y}), Health={before_health}")

        # Execute input for multiple frames
        for frame in range(60):  # Hold for 1 second (60 frames)
            step_result = env.step(input_array)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            if hasattr(env, "render"):
                env.render()
            time.sleep(0.016)  # ~60 FPS

            # Check every 15 frames
            if frame % 15 == 0:
                temp_info = read_ram_values(env, info)
                temp_x = temp_info.get("x_position", before_x)
                temp_y = temp_info.get("y_position", before_y)
                temp_health = temp_info.get("health", before_health)

                if (
                    temp_x != before_x
                    or temp_y != before_y
                    or temp_health != before_health
                ):
                    print(
                        f"   Frame {frame:2d}: Pos=({temp_x}, {temp_y}), Health={temp_health}"
                    )

        # Final position after input
        after_info = read_ram_values(env, info)
        after_x = after_info.get("x_position", before_x)
        after_y = after_info.get("y_position", before_y)
        after_health = after_info.get("health", before_health)

        print(f"   After:  Pos=({after_x}, {after_y}), Health={after_health}")

        # Check for changes
        dx = after_x - before_x
        dy = after_y - before_y
        dh = after_health - before_health

        if dx != 0 or dy != 0:
            print(f"   ✅ POSITION CHANGED: Δx={dx}, Δy={dy}")
        else:
            print(f"   ❌ NO POSITION CHANGE")

        if dh != 0:
            print(f"   💥 HEALTH CHANGED: Δh={dh}")

        # Rest between tests
        print("   💤 Resting...")
        for _ in range(30):
            step_result = env.step(test_inputs["IDLE"])
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            if hasattr(env, "render"):
                env.render()

    print(f"\n🏁 DIAGNOSTIC COMPLETE")
    env.close()


def test_raw_ram_reading():
    """Test if we can read RAM at all"""
    print("\n🔍 RAW RAM DIAGNOSTIC")
    print("=" * 30)

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = retro.make(game=game, players=1)
    env.reset()

    # Test different ways to access RAM
    print("Testing RAM access methods...")

    # Method 1: env.unwrapped.get_ram()
    try:
        if hasattr(env.unwrapped, "get_ram"):
            ram = env.unwrapped.get_ram()
            print(f"✅ Method 1 (env.unwrapped.get_ram): RAM size = {len(ram)}")
            print(f"   Sample bytes: {ram[:10].tolist()}")
        else:
            print("❌ Method 1: No get_ram method")
    except Exception as e:
        print(f"❌ Method 1 error: {e}")

    # Method 2: env.unwrapped.em.get_ram()
    try:
        if hasattr(env.unwrapped, "em") and hasattr(env.unwrapped.em, "get_ram"):
            ram = env.unwrapped.em.get_ram()
            print(f"✅ Method 2 (env.unwrapped.em.get_ram): RAM size = {len(ram)}")
            print(f"   Sample bytes: {ram[:10].tolist()}")
        else:
            print("❌ Method 2: No em.get_ram method")
    except Exception as e:
        print(f"❌ Method 2 error: {e}")

    # Method 3: Check what's available
    print(f"\n🔍 Environment attributes:")
    print(f"   env type: {type(env)}")
    print(f"   env.unwrapped type: {type(env.unwrapped)}")
    if hasattr(env.unwrapped, "em"):
        print(f"   env.unwrapped.em type: {type(env.unwrapped.em)}")
        em_attrs = [attr for attr in dir(env.unwrapped.em) if not attr.startswith("_")]
        print(f"   em methods: {em_attrs[:10]}...")  # Show first 10

    env.close()


if __name__ == "__main__":
    print("🚀 STREET FIGHTER MOVEMENT DIAGNOSTIC")
    print("This will test if movement inputs work at all")
    print("Watch the screen - the character should move!")
    print("\nPress Enter to start...")
    input()

    # First test raw RAM access
    test_raw_ram_reading()

    print("\nPress Enter to start movement test...")
    input()

    # Then test movement
    test_movement()
