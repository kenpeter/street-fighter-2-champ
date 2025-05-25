import retro
import os
import time
import numpy as np


def scan_for_position_addresses():
    """Scan RAM to find the actual player position addresses"""
    print("🔍 SCANNING FOR CORRECT POSITION ADDRESSES")
    print("=" * 50)

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

    # Let game stabilize
    NO_ACTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for _ in range(30):
        step_result = env.step(NO_ACTION)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated

    # Get initial RAM snapshot
    ram_before = env.unwrapped.get_ram().copy()
    print(f"📸 Initial RAM snapshot taken ({len(ram_before)} bytes)")

    # Execute RIGHT movement for 60 frames
    RIGHT = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    print("➡️  Executing RIGHT movement for 60 frames...")

    for i in range(60):
        step_result = env.step(RIGHT)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        env.render()
        time.sleep(0.016)

    # Get RAM after movement
    ram_after = env.unwrapped.get_ram().copy()
    print("📸 Post-movement RAM snapshot taken")

    # Find addresses that changed
    changed_addresses = []
    for addr in range(len(ram_before)):
        if ram_before[addr] != ram_after[addr]:
            changed_addresses.append(
                {
                    "address": addr,
                    "before": ram_before[addr],
                    "after": ram_after[addr],
                    "diff": ram_after[addr] - ram_before[addr],
                }
            )

    print(f"\n🔄 Found {len(changed_addresses)} changed memory addresses")

    # Filter for likely position candidates (reasonable values and changes)
    position_candidates = []
    for change in changed_addresses:
        addr = change["address"]
        before = change["before"]
        after = change["after"]
        diff = change["diff"]

        # Look for reasonable position values (0-400 range for screen positions)
        # and reasonable changes (1-20 pixels movement)
        if 0 <= before <= 400 and 0 <= after <= 400 and 1 <= abs(diff) <= 50:
            position_candidates.append(change)

    print(f"🎯 Found {len(position_candidates)} potential position addresses:")

    # Show top candidates
    for i, candidate in enumerate(position_candidates[:20]):  # Show top 20
        addr = candidate["address"]
        before = candidate["before"]
        after = candidate["after"]
        diff = candidate["diff"]

        # Try to read as 16-bit values too
        if addr + 1 < len(ram_after):
            before_16 = (ram_before[addr] << 8) | ram_before[addr + 1]
            after_16 = (ram_after[addr] << 8) | ram_after[addr + 1]
            diff_16 = after_16 - before_16

            print(
                f"  {i+1:2d}. Address {addr:5d} (0x{addr:04X}): "
                f"{before:3d}→{after:3d} (Δ{diff:+3d}) | "
                f"16-bit: {before_16:3d}→{after_16:3d} (Δ{diff_16:+3d})"
            )
        else:
            print(
                f"  {i+1:2d}. Address {addr:5d} (0x{addr:04X}): "
                f"{before:3d}→{after:3d} (Δ{diff:+3d})"
            )

    # Test the most promising addresses
    print(f"\n🧪 TESTING TOP CANDIDATES...")

    # Reset and test each candidate
    env.em.set_state(state_data)
    for _ in range(30):
        step_result = env.step(NO_ACTION)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result

    # Test movement again and track specific addresses
    if position_candidates:
        top_candidates = position_candidates[:5]  # Test top 5

        print("📍 Before movement:")
        ram_test = env.unwrapped.get_ram()
        for i, candidate in enumerate(top_candidates):
            addr = candidate["address"]
            val = ram_test[addr]
            val_16 = (
                (ram_test[addr] << 8) | ram_test[addr + 1]
                if addr + 1 < len(ram_test)
                else 0
            )
            print(f"  Candidate {i+1}: Addr {addr} = {val} (16-bit: {val_16})")

        # Move right again
        for _ in range(30):
            step_result = env.step(RIGHT)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result

        print("📍 After movement:")
        ram_test = env.unwrapped.get_ram()
        for i, candidate in enumerate(top_candidates):
            addr = candidate["address"]
            val = ram_test[addr]
            val_16 = (
                (ram_test[addr] << 8) | ram_test[addr + 1]
                if addr + 1 < len(ram_test)
                else 0
            )
            print(f"  Candidate {i+1}: Addr {addr} = {val} (16-bit: {val_16})")

    # Also scan for known ranges that might contain positions
    print(f"\n🔍 CHECKING COMMON POSITION ADDRESS RANGES...")

    # Common ranges for game data
    ranges_to_check = [
        (0xFF0000, 0xFF2000, "Work RAM 1"),
        (0xFF8000, 0xFFA000, "Work RAM 2"),
        (0x00A000, 0x00C000, "Game Data 1"),
        (0x010000, 0x012000, "Game Data 2"),
    ]

    for start, end, name in ranges_to_check:
        if start < len(ram_before) and end < len(ram_before):
            print(f"  {name} ({start:06X}-{end:06X}):")
            changes_in_range = [
                c for c in changed_addresses if start <= c["address"] < end
            ]
            if changes_in_range:
                for change in changes_in_range[:5]:  # Show first 5
                    addr = change["address"]
                    print(
                        f"    Addr {addr:5d} (0x{addr:04X}): "
                        f"{change['before']:3d}→{change['after']:3d}"
                    )
            else:
                print(f"    No changes found")

    env.close()

    print(f"\n✅ SCAN COMPLETE!")
    print(f"💡 Look for addresses with consistent small changes (1-10 pixels)")
    print(f"💡 Player X position should increase when moving RIGHT")


if __name__ == "__main__":
    print("🚀 RAM ADDRESS SCANNER")
    print("This will find the CORRECT memory addresses for player position")
    print("Watch the character move RIGHT and we'll find which RAM addresses change!")
    print("\nPress Enter to start...")
    input()

    scan_for_position_addresses()
