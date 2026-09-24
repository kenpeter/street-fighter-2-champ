import argparse
import json
from pathlib import Path

import imageio.v2 as imageio

from laya_sf2 import (
    A_A,
    A_A_DOWN,
    A_B,
    A_C_LEFT,
    A_C_RIGHT,
    A_DOWN,
    A_DOWN_LEFT,
    A_DOWN_RIGHT,
    A_IDLE,
    A_LEFT,
    A_RIGHT,
    A_UP_LEFT,
    A_UP_RIGHT,
    A_X,
    A_X_LEFT,
    A_X_RIGHT,
    A_Y,
    A_Z,
    ART_DEC,
    ART_VID,
    HITSTUN_LIKE,
    THROW_RECOVERY,
    detect_outcome,
    make_env,
    read_state,
    wait_for_round,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "artifacts" / "playthrough" / "sft" / "sf2_video_ken_ryu_4_direct.jsonl"


def load_moves(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row.get("vlm_move", "idle") for row in rows]


def directions(state):
    face_left = int(state["px"]) > int(state["ex"])
    toward = A_LEFT if face_left else A_RIGHT
    away = A_RIGHT if face_left else A_LEFT
    down_away = A_DOWN_RIGHT if face_left else A_DOWN_LEFT
    up_toward = A_UP_LEFT if face_left else A_UP_RIGHT
    return toward, away, down_away, up_toward, face_left


ENEMY_ATTACK = {518, 520, 522, 524, 526, 528, 530}


def threat_state(state):
    return int(state["estatus"]) in ENEMY_ATTACK or bool(state.get("ram_e_proj"))


def guard_state(state):
    return int(state["pstatus"]) in HITSTUN_LIKE or int(state["pstatus"]) in THROW_RECOVERY


def move_segments(move, state, index, setup):
    toward, away, down_away, up_toward, face_left = directions(state)
    distance = abs(int(state["px"]) - int(state["ex"]))
    if guard_state(state) or threat_state(state):
        return [("block", 6)]
    if setup == "tatsu" and distance > 85:
        return [("tatsu_down", 3), ("tatsu_diag", 4), ("tatsu_punch", 3), ("tatsu_finish", 1)]
    if setup == "fireball" and distance > 85:
        return [("fb_down", 2), ("fb_diag", 2), ("fb_punch", 3)]
    if move == "light_attack":
        if distance > 85:
            return [("walk", 3), ("light", 3)]
        return [("light", 3)]
    if move == "crouch_kick":
        return [("crouch_kick", 3)]
    if move == "crouch_block":
        return [("block", 6)]
    if move == "jump_toward":
        return [("jump", 10)]
    if move == "heavy_attack":
        return [("heavy", 4)]
    if move == "hadouken":
        return [("fb_down", 2), ("fb_diag", 2), ("fb_punch", 3)]
    if move == "shoryuken":
        return [("fb_down", 2), ("fb_diag", 2), ("shoryuken", 3)]
    if move == "tatsu":
        return [("tatsu_down", 3), ("tatsu_diag", 4), ("tatsu_punch", 3), ("tatsu_finish", 1)]
    if move in ("idle", "hitstun", "knocked_down"):
        return [("wait", 4)]
    if move == "walk":
        return [("walk", 4)]
    return [("wait", 3)]


def action_for(kind, state, index):
    toward, away, down_away, up_toward, face_left = directions(state)
    if kind == "walk":
        return toward, "walk"
    if kind == "light":
        return (A_B if index % 2 == 0 else A_A), "light"
    if kind == "crouch_kick":
        return A_A_DOWN, "crouch_kick"
    if kind == "block":
        return down_away, "block"
    if kind == "jump":
        return up_toward, "jump"
    if kind == "heavy":
        return A_Z, "heavy"
    if kind == "tatsu_down":
        return A_DOWN, "tatsu_down"
    if kind == "tatsu_diag":
        return (A_DOWN_RIGHT if face_left else A_DOWN_LEFT), "tatsu_diag"
    if kind in ("tatsu_punch", "tatsu_finish"):
        return (A_C_RIGHT if face_left else A_C_LEFT), "tatsu_punch"
    if kind == "fb_down":
        return A_DOWN, "fireball_down"
    if kind == "fb_diag":
        return (A_DOWN_LEFT if face_left else A_DOWN_RIGHT), "fireball_diag"
    if kind == "fb_punch":
        return (A_X_LEFT if face_left else A_X_RIGHT), "fireball_punch"
    if kind == "shoryuken":
        return (A_Y | up_toward), "shoryuken"
    if kind == "down_away":
        return down_away, "special_away"
    return A_IDLE, "wait"


def run(args):
    moves = load_moves(args.data)
    if not moves:
        raise ValueError("replay sequence is empty")
    env = make_env(args.state, render_mode="rgb_array")
    for _ in range(args.delay):
        env.step(A_IDLE)
    wait_for_round(env, max_frames=500)
    initial = read_state(env)
    init_mw = int(initial["matches_won"])
    init_emw = int(initial["enemy_matches_won"])
    init_hp = int(initial["health"])
    init_ehp = int(initial["enemy_health"])
    tag = args.tag
    video_path = Path(ART_VID) / f"sf2_{args.state}_{tag}.mp4"
    log_path = Path(ART_DEC) / f"sf2_{args.state}_{tag}_replay.jsonl"
    result_path = Path(ART_DEC) / f"sf2_{args.state}_{tag}_replay.result.json"
    writer = imageio.get_writer(
        str(video_path), fps=30, codec="libx264", quality=None,
        ffmpeg_params=["-crf", "30", "-preset", "veryfast", "-pix_fmt", "yuv420p"],
    )
    log = log_path.open("w", buffering=1)
    state = initial
    action = A_IDLE
    frames_left = 0
    segment_index = 0
    move_index = 0
    tick = 0
    outcome = None
    rest = 0
    pending = []
    current_move = None
    try:
        while tick < args.max_frames:
            if frames_left <= 0 and rest <= 0:
                if guard_state(state) or threat_state(state):
                    pending = []
                if not pending:
                    current_move = moves[move_index % len(moves)]
                    pending.extend(move_segments(current_move, state, segment_index, args.setup))
                    move_index += 1
                kind, duration = pending.pop(0)
                action, why = action_for(kind, state, segment_index)
                frames_left = duration
                segment_index += 1
                log.write(json.dumps({
                    "t": tick, "move": current_move, "kind": kind, "action": int(action),
                    "why": why, "hp": int(state["health"]), "ehp": int(state["enemy_health"]),
                    "dist": abs(int(state["px"]) - int(state["ex"])),
                }) + "\n")
            elif rest > 0:
                action = A_IDLE
                frames_left = 1
                rest -= 1
            env.step(action)
            tick += 1
            frames_left -= 1
            state = read_state(env)
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)
            outcome_now = detect_outcome(state, init_mw, init_emw, init_hp, init_ehp)
            if outcome_now:
                outcome = outcome_now
                break
            if int(state["health"]) <= 0 or int(state["enemy_health"]) <= 0:
                action = A_IDLE
                frames_left = 0
                rest = 45
                move_index = 0
                pending = []
                current_move = None
        if outcome is None:
            outcome = "TIMEOUT"
    finally:
        writer.close()
        log.close()
        env.close()
    result = {
        "outcome": outcome,
        "ticks": tick,
        "state": args.state,
        "moves": len(moves),
        "hp": int(state["health"]),
        "ehp": int(state["enemy_health"]),
        "mw": int(state["matches_won"]),
        "emw": int(state["enemy_matches_won"]),
        "video": str(video_path),
        "log": str(log_path),
    }
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if outcome == "WIN" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="ken_ryu_4")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--max-frames", type=int, default=6000)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--setup", choices=["tatsu", "fireball", "none"], default="tatsu")
    parser.add_argument("--tag", default="inferred")
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
