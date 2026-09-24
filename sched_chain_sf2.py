"""Chain scheduler: Laya-brain every SF2 opponent 1->12 unattended.

Per opponent: base -> aggressive -> turtle x seeds/delays until WIN
  -> headless verify already implied by outcome WIN
  -> keep sf2_<state>.mp4
  -> reflection log on FAIL, then next attempt with new knobs
  -> chain_status.json resume; stops only after ken_bison_12.

Run:
  /home/kenpeter/work/Mario/.venv-phase2/bin/python sched_chain_sf2.py
Log: /tmp/sf2_chain.log   Status: ./chain_status.json
"""
import json
import os
import sys
import time
import traceback

WORK = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(WORK, "chain_status.json")
ART = os.path.join(WORK, "artifacts")
ART_VID = os.path.join(ART, "videos")
ART_DEC = os.path.join(ART, "decisions")
for _d in (ART, ART_VID, ART_DEC):
    os.makedirs(_d, exist_ok=True)
sys.path.insert(0, WORK)

from laya_sf2 import (  # noqa: E402
    OPP_NAME,
    load_agent,
    run_play,
)

# arcade ladder by state suffix order in repo
STAGES = [
    "ken_chunli_1",
    "ken_zang_2",
    "ken_dah_3",
    "ken_ryu_4",
    "ken_honda_5",
    "ken_blanka_6",
    "ken_gulie_7",
    "ken_ken_8",
    "ken_barog_9",
    "ken_vega_10",
    "ken_sagat_11",
    "ken_bison_12",
]

# Infinite retries on the first unfinished stage (user: do not advance until WIN).
MAX_ATTEMPTS_PER_STAGE = 10**9
MAX_FRAMES = 9000
# attempt grid: (strategy, delay, seed) — aggressive early (near-wins were close-range)
ATTEMPT_GRID = [
    ("aggressive", 0, 777),
    ("aggressive", 15, 1),
    ("aggressive", 0, 1),
    ("base", 0, 777),
    ("aggressive", 0, 2),
    ("base", 30, 1),
    ("aggressive", 30, 2),
    ("turtle", 0, 777),
    ("base", 0, 2),
    ("aggressive", 45, 1),
    ("base", 45, 777),
    ("turtle", 15, 1),
    ("aggressive", 15, 777),
    ("base", 15, 777),
    ("aggressive", 30, 777),
    ("turtle", 0, 1),
    ("base", 30, 2),
    ("aggressive", 45, 2),
]


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open("/tmp/sf2_chain.log", "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_status():
    if os.path.isfile(STATUS):
        with open(STATUS) as f:
            return json.load(f)
    return {"done": [], "failed": [], "reflections": {}}


def load_status_resume():
    st = load_status()
    st.setdefault("done", [])
    st.setdefault("failed", [])
    st.setdefault("reflections", {})
    st.setdefault("active", None)
    st["failed"] = []
    save_status(st)
    return st


def save_status(st):
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATUS)


def reflect(stage, attempt, final, strat, delay, seed):
    """Build a short reflection string used to pick the next attempt."""
    hp = final.get("hp", 0)
    ehp = final.get("ehp", 0)
    mw, emw = final.get("mw", 0), final.get("emw", 0)
    tail = final.get("history_tail") or []
    last_whys = [h.split(" ")[1] if len(h.split(" ")) > 1 else h for h in tail[-6:]]
    if final.get("outcome") == "QUIT":
        reason = "quit"
    elif hp <= 0:
        reason = "died (hp=0)"
    elif emw >= 2:
        reason = f"lost rounds {mw}-{emw}"
    elif ehp < hp:
        reason = f"time/round-end behind on damage hp={hp} ehp={ehp}"
    elif final.get("ticks", 0) >= MAX_FRAMES - 50:
        reason = f"timeout ticks={final.get('ticks')} hp={hp} ehp={ehp}"
    else:
        reason = f"no-win hp={hp} ehp={ehp} rd={mw}-{emw}"
    refl = (
        f"attempt{attempt} strat={strat} d={delay} s={seed}: {reason}; "
        f"last_actions={last_whys}"
    )
    # self-improvement knobs from damage/why pattern (not just hp)
    suggestion = "vary-seed-delay"
    last_w = " ".join(last_whys)
    if ehp <= 40 and hp > 0:
        suggestion = "close-and-finish"  # almost had them — pressure to kill
    elif ehp >= 120 and hp <= 40:
        suggestion = "more-damage-zone"  # barely scratched them — fireball/approach
    elif ehp >= 100:
        suggestion = "more-damage-zone"
    elif hp < 40 and ehp < 80:
        suggestion = "more-blocking"  # traded down — survive longer
    elif final.get("ticks", 0) >= MAX_FRAMES - 50 and ehp > 80:
        suggestion = "close-gap-more"
    # death while throw-escape was spam => jump out more, don't sit in throw range
    if any("throw-escape" in w for w in last_whys) and hp <= 0:
        suggestion = "jump-more-throw"
    # dying while AA-block spamming / finish-mash: bias next attempt to survive finish
    if hp <= 0 and ("aa-block" in last_w or "finish-" in last_w) and ehp < 90:
        suggestion = "more-blocking"
    return refl, suggestion


def next_attempt(attempt, suggestion):
    """Pick grid slot with light self-improvement bias."""
    if attempt < len(ATTEMPT_GRID):
        strat, delay, seed = ATTEMPT_GRID[attempt]
        if suggestion == "more-blocking" and strat != "turtle":
            strat = "turtle"
        if suggestion in ("close-and-finish", "close-gap-more", "more-damage-zone") and strat == "turtle":
            strat = "aggressive"
        # bump seed on repeats
        if attempt >= len(ATTEMPT_GRID):
            seed = 777 + attempt
        return strat, delay, seed
    # beyond grid: random-ish unique
    strats = ["base", "aggressive", "turtle"]
    strat = strats[attempt % 3]
    if suggestion == "more-blocking":
        strat = "turtle"
    elif suggestion in ("more-aggressive-or-better-punish", "close-gap-more", "more-damage-zone", "close-and-finish", "jump-more-throw"):
        strat = "aggressive"
    delay = (attempt * 15) % 60
    seed = 1000 + attempt * 17
    return strat, delay, seed


def _find(base, name):
    for p in (os.path.join(base, name), os.path.join(WORK, name),
              os.path.join(ART_VID, name), os.path.join(ART_DEC, name)):
        if os.path.isfile(p) and os.path.getsize(p) > 10000:
            return p
    return None


def stage_video(stage):
    return _find(WORK, f"sf2_{stage}.mp4") or _find(ART_VID, f"sf2_{stage}.mp4")


def active_attempt(st, stage):
    active = st.get("active") or {}
    if active.get("stage") != stage:
        return None
    try:
        attempt = int(active["attempt"])
        delay = int(active["delay"])
        seed = int(active["seed"])
        max_frames = int(active.get("max_frames", MAX_FRAMES))
    except (KeyError, TypeError, ValueError):
        return None
    strategy = active.get("strategy")
    if attempt < 0 or max_frames <= 0 or strategy not in ("base", "aggressive", "turtle"):
        return None
    return attempt, strategy, delay, seed, max_frames


def do_stage(stage, agent, st, lora=None):
    log(f"{stage} ({OPP_NAME.get(stage, stage)}): start lora={lora is not None}")
    while True:
        resumed = active_attempt(st, stage)
        if resumed is None:
            refl_prev = (st.get("reflections") or {}).get(stage, {})
            try:
                previous_attempt = int(refl_prev.get("attempt", -1))
            except (TypeError, ValueError):
                previous_attempt = -1
            attempt = max(0, previous_attempt + 1)
            suggestion = refl_prev.get("suggestion", "vary-seed-delay")
            strat, delay, seed = next_attempt(attempt, suggestion)
            max_frames = MAX_FRAMES
        else:
            attempt, strat, delay, seed, max_frames = resumed
        if attempt >= MAX_ATTEMPTS_PER_STAGE:
            log(f"{stage}: FAILED after unexpected attempt cap")
            return False
        tag = f"{stage}_a{attempt}"
        st["active"] = {
            "stage": stage,
            "attempt": attempt,
            "strategy": strat,
            "delay": delay,
            "seed": seed,
            "max_frames": max_frames,
            "tag": tag,
        }
        save_status(st)
        log(
            f"{stage} attempt {attempt + 1} "
            f"strat={strat} delay={delay} seed={seed}"
        )
        try:
            outcome, ticks, final = run_play(
                stage,
                max_frames=max_frames,
                seed=seed,
                delay=delay,
                tag=tag,
                agent=agent,
                live=False,
                strategy=strat,
                record=True,
                lora=lora,
            )
        except Exception:
            log(f"{stage}: EXCEPTION\n{traceback.format_exc()}")
            final = {"outcome": "ERROR", "ticks": 0}
            outcome = None

        if outcome == "WIN":
            # promote video to canonical name (root + artifacts)
            src = os.path.join(WORK, f"sf2_{tag}.mp4")
            if not os.path.isfile(src):
                src = os.path.join(ART_VID, f"sf2_{tag}.mp4")
            for dst in (
                os.path.join(WORK, f"sf2_{stage}.mp4"),
                os.path.join(ART_VID, f"sf2_{stage}.mp4"),
            ):
                if os.path.isfile(src):
                    try:
                        import shutil
                        shutil.copy2(src, dst)
                    except OSError:
                        pass
                # prefer ART_VID copy as canonical src for next hop
                if os.path.isfile(dst):
                    src = dst
            # keep attempt recording as well (win archive)
            src_att = os.path.join(ART_VID, f"sf2_{tag}.mp4")
            if os.path.isfile(src_att):
                try:
                    import shutil
                    shutil.copy2(src_att, os.path.join(ART_VID, f"sf2_{stage}_win_a{attempt}.mp4"))
                except OSError:
                    pass
            # also promote decisions log
            src_l = os.path.join(WORK, f"sf2_decisions_{tag}.jsonl")
            if not os.path.isfile(src_l):
                src_l = os.path.join(ART_DEC, f"sf2_decisions_{tag}.jsonl")
            for dst_l in (
                os.path.join(WORK, f"sf2_decisions_{stage}.jsonl"),
                os.path.join(ART_DEC, f"sf2_decisions_{stage}.jsonl"),
            ):
                if os.path.isfile(src_l):
                    try:
                        os.replace(src_l, dst_l)
                    except OSError:
                        import shutil
                        shutil.copy2(src_l, dst_l)
                    src_l = dst_l
            v = stage_video(stage)
            log(f"{stage}: WIN ticks={ticks} video={v}")
            st.setdefault("reflections", {})[stage] = {
                "outcome": "WIN",
                "attempt": attempt,
                "strategy": strat,
                "delay": delay,
                "seed": seed,
                "ticks": ticks,
                "video": v,
            }
            st["active"] = None
            if stage not in st["done"]:
                st["done"].append(stage)
            save_status(st)
            return True

        refl, suggestion = reflect(stage, attempt, final, strat, delay, seed)
        log(f"{stage}: FAIL ({final.get('outcome')}) {refl} -> next={suggestion}")
        # win-only videos: drop this attempt's recording
        for vp in (
            os.path.join(WORK, f"sf2_{tag}.mp4"),
            os.path.join(ART_VID, f"sf2_{tag}.mp4"),
        ):
            try:
                if os.path.isfile(vp):
                    os.remove(vp)
            except OSError:
                pass
        st.setdefault("reflections", {})[stage] = {
            "outcome": final.get("outcome"),
            "attempt": attempt,
            "reflection": refl,
            "suggestion": suggestion,
            "strategy": strat,
            "delay": delay,
            "seed": seed,
            "hp": final.get("hp"),
            "ehp": final.get("ehp"),
            "mw": final.get("mw"),
            "emw": final.get("emw"),
            "ticks": final.get("ticks"),
        }
        st["active"] = None
        save_status(st)
        attempt += 1


def main():
    st = load_status_resume()
    log(f"chain start done={st['done']} failed={st.get('failed', [])} (single-stage lock)")
    agent = load_agent()
    lora = None  # LoRA removed from play path — RAM + Laya + rules only
    log("LoRA disabled for play; feeding full RAM to Laya")
    # single-stage lock: only first not-done stage; infinite retries until WIN
    while True:
        active = st.get("active") or {}
        active_stage = active.get("stage")
        if active_stage not in STAGES or active_stage in st["done"]:
            if active_stage is not None:
                st["active"] = None
                save_status(st)
            active_stage = None
        stage = active_stage or next((s for s in STAGES if s not in st["done"]), None)
        if stage is None:
            break
        if not stage_video(stage):
            if stage in st["done"]:
                log(f"{stage}: marked done but missing video, retrying")
                st["done"].remove(stage)
                save_status(st)
                continue
        try:
            if do_stage(stage, agent, st, lora=None):
                if stage not in st["done"]:
                    st["done"].append(stage)
                if stage in st.get("failed", []):
                    st["failed"].remove(stage)
                save_status(st)
            else:
                # keep retrying same stage — never advance on failure
                log(f"{stage}: no WIN yet, holding stage (will retry)")
                if stage not in st.get("failed", []):
                    st.setdefault("failed", []).append(stage)
                save_status(st)
        except Exception:
            log(f"{stage}: EXCEPTION\n{traceback.format_exc()}")
            st.setdefault("failed", []).append(stage)
            save_status(st)
    log(
        f"CHAIN COMPLETE done={st['done']} failed={st.get('failed', [])} "
        f"boss_beaten={'ken_bison_12' in st['done']}"
    )


if __name__ == "__main__":
    main()
