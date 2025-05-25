# COMPREHENSIVE MOVEMENT DIAGNOSTIC
# Add this to your eval.py to diagnose the exact problem


def diagnose_movement_problem(agent, model_path):
    """Comprehensive diagnosis of why agent won't move"""
    print("🔍 MOVEMENT PROBLEM DIAGNOSIS")
    print("=" * 50)

    # Step 1: Test if the model is making movement decisions
    print("STEP 1: Testing Agent Decision Making")
    print("-" * 40)

    test_scenarios = [
        {"name": "Far from enemy", "x_position": 50, "enemy_x_position": 200},
        {"name": "Close to enemy", "x_position": 180, "enemy_x_position": 200},
        {"name": "Behind enemy", "x_position": 220, "enemy_x_position": 200},
    ]

    for scenario in test_scenarios:
        print(f"\n📍 Scenario: {scenario['name']}")
        print(
            f"   Player at {scenario['x_position']}, Enemy at {scenario['enemy_x_position']}"
        )

        # Create test info
        info = {
            "health": 176,
            "enemy_health": 176,
            "x_position": scenario["x_position"],
            "enemy_x_position": scenario["enemy_x_position"],
            "y_position": 0,
            "enemy_y_position": 0,
            "status": 512,
            "enemy_status": 512,
            "matches_won": 0,
            "enemy_matches_won": 0,
            "enemy_character": 0,
            "score": 0,
        }

        # Get agent's decision
        dummy_obs = np.zeros((224, 256, 3))
        action_index, frame_inputs = agent.getMove(dummy_obs, info)
        move_name = list(agent.moveList)[action_index].name

        print(f"   Agent decision: {move_name} (action {action_index})")
        print(f"   Frame inputs: {frame_inputs}")

        # Check if it's a movement action
        movement_actions = [
            "Right",
            "Left",
            "Up",
            "Down",
            "UpRight",
            "UpLeft",
            "DownRight",
            "DownLeft",
        ]
        if move_name in movement_actions:
            print(f"   ✅ Agent chose MOVEMENT action!")
        else:
            print(f"   ❌ Agent chose NON-MOVEMENT action")

    # Step 2: Test the Q-values for movement actions
    print(f"\nSTEP 2: Q-Value Analysis for Movement Actions")
    print("-" * 50)

    info = {
        "health": 176,
        "enemy_health": 176,
        "x_position": 50,
        "enemy_x_position": 200,
        "y_position": 0,
        "enemy_y_position": 0,
        "status": 512,
        "enemy_status": 512,
        "matches_won": 0,
        "enemy_matches_won": 0,
        "enemy_character": 0,
        "score": 0,
    }

    stateData = agent.prepareNetworkInputs(info)
    predictedRewards = agent.model.predict(stateData, verbose=0)[0]

    print("Q-values for all actions:")
    for i, move in enumerate(agent.moveList):
        q_val = predictedRewards[i]
        move_name = move.name
        move_type = (
            "MOVEMENT" if move_name in ["Right", "Left", "Up", "Down"] else "OTHER"
        )
        print(f"   {i:2d}: {move_name:15s} = {q_val:8.4f} ({move_type})")

    # Check if movement actions have reasonable Q-values
    movement_indices = []
    for i, move in enumerate(agent.moveList):
        if move.name in ["Right", "Left"]:
            movement_indices.append(i)

    if movement_indices:
        avg_movement_q = np.mean([predictedRewards[i] for i in movement_indices])
        max_q = np.max(predictedRewards)
        print(f"\nMovement action Q-values: {avg_movement_q:.4f} (avg)")
        print(f"Maximum Q-value: {max_q:.4f}")

        if avg_movement_q < max_q * 0.5:
            print("❌ PROBLEM: Movement Q-values are much lower than other actions!")
            print("   This suggests the agent learned NOT to move.")
        else:
            print("✅ Movement Q-values seem reasonable")

    # Step 3: Test raw environment movement
    print(f"\nSTEP 3: Testing Raw Environment Movement")
    print("-" * 50)

    test_environment_movement()

    # Step 4: Check frame input generation
    print(f"\nSTEP 4: Frame Input Generation Test")
    print("-" * 50)

    for move in [agent.moveList(1), agent.moveList(5)]:  # Right and Left
        if hasattr(agent.moveList, "__call__"):
            continue
        move_name = move.name
        frame_inputs = agent.convertMoveToFrameInputs(move, info)
        print(f"{move_name:10s}: {frame_inputs}")

        # Verify frame inputs are correct
        if move_name == "Right" and frame_inputs == [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
        ]:
            print(f"   ✅ {move_name} frame inputs are CORRECT")
        elif move_name == "Left" and frame_inputs == [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ]:
            print(f"   ✅ {move_name} frame inputs are CORRECT")
        else:
            print(f"   ❌ {move_name} frame inputs might be WRONG")


def test_environment_movement():
    """Test if the environment itself supports movement"""
    try:
        game = "StreetFighterIISpecialChampionEdition-Genesis"
        env = retro.make(game=game, players=1)
        env.reset()

        # Load a state if available
        state_dir = "./StreetFighterIISpecialChampionEdition-Genesis"
        if os.path.exists(state_dir):
            state_files = [f for f in os.listdir(state_dir) if f.endswith(".state")]
            if state_files:
                state_path = os.path.join(state_dir, state_files[0])
                with open(state_path, "rb") as f:
                    state_data = f.read()
                env.em.set_state(state_data)

        print("Testing basic movement commands...")

        # Test movements
        movements = [
            ("RIGHT", [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            ("LEFT", [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        ]

        NO_ACTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for direction, action in movements:
            # Stabilize
            for _ in range(5):
                env.step(NO_ACTION)

            # Get initial position
            obs, _, _, info = env.step(NO_ACTION)
            initial_x = info.get("x_position", 100)
            print(f"   {direction}: Initial X = {initial_x}")

            # Execute movement
            for _ in range(15):  # Hold for more frames
                obs, _, _, info = env.step(action)

            final_x = info.get("x_position", 100)
            print(
                f"   {direction}: Final X = {final_x}, Change = {final_x - initial_x}"
            )

            if abs(final_x - initial_x) > 0:
                print(f"   ✅ {direction} movement works in environment!")
            else:
                print(f"   ❌ {direction} movement failed in environment!")

        env.close()

    except Exception as e:
        print(f"❌ Environment test failed: {e}")


def check_action_space_mismatch(agent):
    """Check if there's a mismatch between trained and eval action spaces"""
    print(f"\nSTEP 5: Action Space Consistency Check")
    print("-" * 50)

    print(f"Current action space size: {agent.actionSize}")
    print(f"Available moves: {len(list(agent.moveList))}")

    if agent.actionSize != len(list(agent.moveList)):
        print("❌ CRITICAL: Action space size mismatch!")
        print("   The model was trained with different number of actions!")
        return False

    print("Available actions:")
    for i, move in enumerate(agent.moveList):
        print(f"   {i}: {move.name}")

    return True


# MAIN DIAGNOSTIC FUNCTION - Call this instead of normal evaluation
def run_full_diagnosis(model_path):
    """Run complete movement diagnosis"""
    print("🔬 COMPREHENSIVE MOVEMENT DIAGNOSIS")
    print("=" * 60)

    try:
        # Load agent
        agent = EvalAgent(model_path=model_path, stateSize=60)
        print("✅ Agent loaded successfully")

        # Run all diagnostic steps
        diagnose_movement_problem(agent, model_path)
        check_action_space_mismatch(agent)

        print(f"\n🎯 DIAGNOSIS COMPLETE")
        print("=" * 60)

        print("LIKELY CAUSES OF NO MOVEMENT:")
        print("1. Agent learned NOT to move (Q-values favor non-movement)")
        print("2. Frame execution timing too short")
        print("3. Action space mismatch between training/eval")
        print("4. State loading puts agent in restricted position")
        print("5. Reward system didn't encourage movement during training")

        print(f"\nRECOMMENDED FIXES:")
        print("1. Use the fixed eval.py with longer frame holds")
        print("2. Retrain with movement rewards")
        print("3. Check training logs for action distribution")
        print("4. Test with different game states")

    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")


# TO USE: Replace your eval.py main section with:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose Street Fighter AI movement issues"
    )
    parser.add_argument(
        "--model_path", type=str, default="models/DeepQAgentModel_300000.weights.h5"
    )
    args = parser.parse_args()

    # Run diagnosis instead of normal evaluation
    run_full_diagnosis(args.model_path)

    # Then run normal evaluation with fixes
    # run_evaluation(model_path=args.model_path, num_games=3, render=True)
