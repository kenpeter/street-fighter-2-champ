import retro
import os
import time


def test_position_addresses():
    """Test the most promising position addresses"""
    print("🧪 TESTING POSITION ADDRESSES")
    print("=" * 40)

    # Candidates from the scan
    candidates = [
        {"addr": 32774, "name": "Candidate A (0x8006)", "desc": "205→217 (+12)"},
        {"addr": 32802, "name": "Candidate B (0x8022)", "desc": "102→128 (+26)"},
        {"addr": 32984, "name": "Candidate C (0x80D8)", "desc": "205→216 (+11)"},
    ]

    # Create environment
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = retro.make(game=game, players=1)
    env.reset()

    # Load state
    state_path = os.path.join(
        os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis"),
        "ken_bison_12.state",
    )
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state_data = f.read()
        env.em.set_state(state_data)

    # Test movements
    movements = [
        {"name": "IDLE", "input": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "frames": 30},
        {"name": "RIGHT", "input": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], "frames": 60},
        {"name": "LEFT", "input": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], "frames": 60},
        {"name": "RIGHT", "input": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], "frames": 30},
    ]

    print("📍 TRACKING POSITION CHANGES:")
    print("Expected: RIGHT should increase X, LEFT should decrease X")
    print()

    for movement in movements:
        print(f"🎮 {movement['name']} for {movement['frames']} frames:")

        # Read before
        ram_before = env.unwrapped.get_ram()
        before_values = {}
        for candidate in candidates:
            addr = candidate["addr"]
            before_values[addr] = ram_before[addr]

        # Execute movement
        for _ in range(movement["frames"]):
            step_result = env.step(movement["input"])
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
            env.render()
            time.sleep(0.008)  # Faster for testing

        # Read after
        ram_after = env.unwrapped.get_ram()

        # Compare
        for candidate in candidates:
            addr = candidate["addr"]
            before = before_values[addr]
            after = ram_after[addr]
            change = after - before

            print(f"  {candidate['name']}: {before:3d} → {after:3d} (Δ{change:+3d})")

        print()

    # Test the most consistent one
    print("🎯 FINAL VERIFICATION:")
    print("The correct player X address should:")
    print("- Increase when moving RIGHT")
    print("- Decrease when moving LEFT")
    print("- Stay roughly the same when IDLE")

    env.close()


def create_fixed_agent():
    """Create a simple test with corrected addresses"""
    print("\n🛠️  CREATING CORRECTED POSITION READER")
    print("=" * 40)

    # Based on the scan, let's test address 32774 (most promising)
    test_addresses = {
        "player_x": 32774,  # 0x8006 - showed 205→217 (+12)
        "player_y": 32775,  # Try next address for Y
        "enemy_x": 32984,  # 0x80D8 - showed 205→216 (+11)
        "enemy_y": 32985,  # Try next address for Y
    }

    def read_positions(env):
        ram = env.unwrapped.get_ram()
        return {
            "player_x": ram[test_addresses["player_x"]],
            "player_y": ram[test_addresses["player_y"]],
            "enemy_x": ram[test_addresses["enemy_x"]],
            "enemy_y": ram[test_addresses["enemy_y"]],
        }

    # Test it
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    env = retro.make(game=game, players=1)
    env.reset()

    state_path = os.path.join(
        os.path.abspath("./StreetFighterIISpecialChampionEdition-Genesis"),
        "ken_bison_12.state",
    )
    if os.path.exists(state_path):
        with open(state_path, "rb") as f:
            state_data = f.read()
        env.em.set_state(state_data)

    # Initialize
    NO_ACTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for _ in range(30):
        step_result = env.step(NO_ACTION)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result

    print("🧪 Testing corrected position reading:")

    # Test sequence
    movements = [
        ("START", NO_ACTION, 1),
        ("RIGHT", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 30),
        ("LEFT", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 60),
        ("RIGHT", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 30),
    ]

    for name, action, frames in movements:
        positions = read_positions(env)
        print(
            f"{name:6s}: Player=({positions['player_x']:3d},{positions['player_y']:3d}) "
            f"Enemy=({positions['enemy_x']:3d},{positions['enemy_y']:3d})"
        )

        for _ in range(frames):
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result

    env.close()

    print(f"\n💡 If player X changes correctly, use these addresses:")
    print(f"   player_x_position: {test_addresses['player_x']}")
    print(f"   player_y_position: {test_addresses['player_y']}")
    print(f"   enemy_x_position: {test_addresses['enemy_x']}")
    print(f"   enemy_y_position: {test_addresses['enemy_y']}")


if __name__ == "__main__":
    print("🚀 POSITION ADDRESS VERIFICATION")
    print("This will test the most promising addresses from the scan")
    print("\nPress Enter to start...")
    input()

    test_position_addresses()
    create_fixed_agent()
