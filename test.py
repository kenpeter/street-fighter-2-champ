import retro
import os
import time
import numpy as np
from DeepQAgent import DeepQAgent


def test_raw_movement():
    """Test if movement works AT ALL in the game"""
    print("🧪 TESTING RAW MOVEMENT IN GAME")
    print("=" * 50)

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

    print("👀 WATCH THE SCREEN - Does Ken move visually?")
    print("If Ken doesn't move on screen, then movement is broken!")

    movements = [
        ("RIGHT", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ("LEFT", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        ("RIGHT", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    ]

    # Initialize
    NO_ACTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for _ in range(60):
        env.step(NO_ACTION)
        env.render()
        time.sleep(0.016)

    print("Now testing movement...")

    for name, action in movements:
        print(f"\n🎮 Testing {name} - WATCH SCREEN!")

        for frame in range(120):  # Hold for 2 seconds
            env.step(action)
            env.render()
            time.sleep(0.016)

        # Rest between movements
        for _ in range(60):
            env.step(NO_ACTION)
            env.render()
            time.sleep(0.016)

    env.close()

    response = input(
        "\n❓ Did you see Ken move LEFT and RIGHT on the screen? (y/n): "
    ).lower()
    return response == "y"


def test_ram_reading_all_addresses():
    """Test EVERY possible address to find real position"""
    print("\n🔍 TESTING ALL RAM ADDRESSES FOR POSITION")
    print("=" * 50)

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

    # Get initial RAM
    NO_ACTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for _ in range(60):
        env.step(NO_ACTION)

    ram_before = env.unwrapped.get_ram().copy()
    print(f"📸 Captured initial RAM state")

    # Move RIGHT for long time
    RIGHT = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    print("➡️  Moving RIGHT for 3 seconds...")
    for _ in range(180):  # 3 seconds
        env.step(RIGHT)
        env.render()
        time.sleep(0.016)

    ram_after = env.unwrapped.get_ram().copy()
    print(f"📸 Captured post-movement RAM state")

    # Find ALL addresses that changed
    changes = []
    for addr in range(len(ram_before)):
        if ram_before[addr] != ram_after[addr]:
            changes.append(
                {
                    "addr": addr,
                    "before": ram_before[addr],
                    "after": ram_after[addr],
                    "diff": int(ram_after[addr]) - int(ram_before[addr]),
                }
            )

    print(f"\n🔄 Found {len(changes)} changed addresses")

    # Look for position-like changes
    position_candidates = []
    for change in changes:
        addr = change["addr"]
        before = change["before"]
        after = change["after"]
        diff = change["diff"]

        # Look for reasonable position changes (1-50 pixel movement)
        if 1 <= abs(diff) <= 50 and 0 <= before <= 400 and 0 <= after <= 400:
            position_candidates.append(change)

    print(f"🎯 Found {len(position_candidates)} position candidates:")
    for i, candidate in enumerate(position_candidates[:20]):
        addr = candidate["addr"]
        before = candidate["before"]
        after = candidate["after"]
        diff = candidate["diff"]
        print(
            f"  {i+1:2d}. Address {addr:5d} (0x{addr:04X}): {before:3d}→{after:3d} (Δ{diff:+3d})"
        )

    env.close()
    return position_candidates


def test_agent_training_data():
    """Test what data the agent actually trained on"""
    print("\n🧠 TESTING AGENT'S TRAINING DATA")
    print("=" * 50)

    # Test what the agent sees vs reality
    agent = DeepQAgent(stateSize=60, total_timesteps=1000)

    # Create fake game scenarios
    scenarios = [
        {"name": "Close", "px": 180, "py": 0, "ex": 200, "ey": 0},
        {"name": "Far", "px": 100, "py": 0, "ex": 300, "ey": 0},
        {"name": "Very Far", "px": 50, "py": 0, "ex": 350, "ey": 0},
    ]

    print("Testing agent's state processing:")
    for scenario in scenarios:
        info = {
            "health": 176,
            "enemy_health": 176,
            "x_position": scenario["px"],
            "y_position": scenario["py"],
            "enemy_x_position": scenario["ex"],
            "enemy_y_position": scenario["ey"],
            "status": 512,
            "enemy_status": 512,
            "matches_won": 0,
            "enemy_matches_won": 0,
            "enemy_character": 0,
            "score": 0,
        }

        distance = abs(scenario["ex"] - scenario["px"])
        print(f"\n📍 {scenario['name']} scenario (distance={distance}):")

        # Get agent's decision
        stateData = agent.prepareNetworkInputs(info)
        predictedRewards = agent.model.predict(stateData, verbose=0)[0]

        # Check movement vs attack preferences
        right_q = predictedRewards[1]  # Right
        left_q = predictedRewards[5]  # Left
        punch_q = predictedRewards[11]  # HeavyPunch
        fireball_q = predictedRewards[25]  # Fireball

        print(f"   Right Q-value: {right_q:.3f}")
        print(f"   Left Q-value: {left_q:.3f}")
        print(f"   HeavyPunch Q-value: {punch_q:.3f}")
        print(f"   Fireball Q-value: {fireball_q:.3f}")

        max_movement = max(right_q, left_q)
        max_attack = max(punch_q, fireball_q)

        if max_attack > max_movement:
            print(
                f"   ❌ Prefers ATTACK ({max_attack:.3f}) over MOVEMENT ({max_movement:.3f})"
            )
        else:
            print(
                f"   ✅ Prefers MOVEMENT ({max_movement:.3f}) over ATTACK ({max_attack:.3f})"
            )


def main():
    print("🚨 COMPREHENSIVE MOVEMENT DEBUG")
    print("This will test EVERYTHING to find why the agent won't move!")

    print("\n1️⃣ First, let's test if movement works in the game at all...")
    input("Press Enter to start visual movement test...")

    movement_works = test_raw_movement()

    if not movement_works:
        print("\n❌ MOVEMENT DOESN'T WORK IN THE GAME!")
        print("The problem is with the game/emulator, not the agent!")
        print("Check:")
        print("- Is the right ROM loaded?")
        print("- Are controls mapped correctly?")
        print("- Is the game state correct?")
        return

    print("\n✅ Movement works visually!")
    print("\n2️⃣ Now let's find the correct RAM addresses...")
    input("Press Enter to scan RAM addresses...")

    candidates = test_ram_reading_all_addresses()

    if not candidates:
        print("\n❌ NO POSITION ADDRESSES FOUND!")
        print("The RAM reading is completely broken!")
        return

    print(f"\n✅ Found {len(candidates)} position candidates!")
    print("\n3️⃣ Finally, let's test what the agent learned...")
    input("Press Enter to analyze agent behavior...")

    test_agent_training_data()

    print("\n🎯 DIAGNOSIS:")
    if movement_works and candidates:
        print("✅ Game movement works")
        print("✅ RAM addresses found")
        print("❌ Agent learned wrong strategy")
        print("\n💡 SOLUTION: Train longer with movement rewards!")
        print("The agent needs to learn that movement → better rewards")
    else:
        print("❌ Fundamental issues found")
        print("Need to fix game/RAM issues first")


if __name__ == "__main__":
    main()
