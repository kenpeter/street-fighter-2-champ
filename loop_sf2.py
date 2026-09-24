"""Closed loop: train -> eval -> reflect -> change -> train ... until chain 12/12.

Stages:
  model:  train vN, eval, write reflections/loop_model.json, apply next train knobs
  play:   smoke a stage with --lora; reflect on decision log
  chain:  run sched_chain_sf2 (LoRA on) until 12/12 or stage stall -> back to model

Run:
  /home/kenpeter/work/Mario/.venv-phase2/bin/python loop_sf2.py [--max-model-iters 8] [--from-stage model|play|chain]
Log: /tmp/sf2_loop.log
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = "/home/kenpeter/work/Mario/.venv-phase2/bin/python"
OUT = ROOT / "lora_sf2_laya"
REFL = ROOT / "reflections"
CHAIN_LOG = Path("/tmp/sf2_chain.log")
LOOP_LOG = Path("/tmp/sf2_loop.log")
STAGES = [
    "ken_chunli_1", "ken_zang_2", "ken_dah_3", "ken_ryu_4",
    "ken_honda_5", "ken_blanka_6", "ken_gulie_7", "ken_ken_8",
    "ken_barog_9", "ken_vega_10", "ken_sagat_11", "ken_bison_12",
]
# thresholds to leave model loop for play smoke
GOALS = {
    "val_top1_min": 0.28,
    "lift_min": 0.08,
    "collapse_max": 0.35,
    "min_classes_with_pred": 4,
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    for p in (LOOP_LOG,):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def run(cmd: list[str], timeout: int | None = None, cwd: Path = ROOT, stream_to: Path | None = None) -> int:
    log("exec: " + " ".join(cmd))
    try:
        if stream_to is not None:
            stream_to.parent.mkdir(parents=True, exist_ok=True)
            fh = stream_to.open("w")
            r = subprocess.run(cmd, cwd=str(cwd), timeout=timeout, text=True,
                               stdout=fh, stderr=subprocess.STDOUT)
            fh.close()
            try:
                tail = "\n".join(stream_to.read_text().splitlines()[-30:])
                if tail:
                    log(f"tail {stream_to}:\n{tail}")
            except OSError:
                pass
            return r.returncode
        r = subprocess.run(cmd, cwd=str(cwd), timeout=timeout, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = r.stdout or ""
        tail = "\n".join(out.splitlines()[-40:])
        if tail:
            log("tail:\n" + tail)
        return r.returncode
    except subprocess.TimeoutExpired as e:
        log(f"TIMEOUT after {timeout}s")
        if e.stdout:
            log("\n".join(str(e.stdout).splitlines()[-20:]))
        return 124


def archive_run(tag: str) -> None:
    if not OUT.exists():
        return
    dest = ROOT / f"lora_sf2_laya_{tag}"
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(OUT, dest)
    log(f"archived -> {dest}")


def load_eval() -> dict | None:
    p = OUT / "eval_result.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def reflect_model(it: int, metrics: dict | None, prev: dict | None) -> dict:
    """Decide next train knobs + whether goals are met."""
    r = {"iter": int(it), "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": metrics}
    if metrics is None:
        r.update({"ok_model": False, "reason": "no eval_result.json", "next": {"epochs": 4}})
        return r
    pred_n = sum((metrics.get("pred_hist") or {}).values())
    n_classes_pred = sum(1 for v in (metrics.get("pred_hist") or {}).values() if v > 0)
    by = metrics.get("by_class") or {}
    weak = sorted(
        [(a, v["acc"], v["n"]) for a, v in by.items() if v.get("n") and v["acc"] is not None and v["acc"] < 0.15 and v["n"] >= 40],
        key=lambda x: x[1],
    )
    ok = (
        metrics.get("val_top1", 0) >= GOALS["val_top1_min"]
        and metrics.get("lift", 0) >= GOALS["lift_min"]
        and metrics.get("collapse", 1) <= GOALS["collapse_max"]
        and n_classes_pred >= GOALS["min_classes_with_pred"]
    )
    # next knobs: stepwise interventions when not ok
    nxt = {"hard_w": 1.0, "soft_w": 0.35, "pr_w": 0.15, "cap": 8.0, "epochs": 4, "lr": 2e-4, "r": 16}
    reasons = []
    if not ok:
        if metrics.get("collapse", 0) > GOALS["collapse_max"]:
            reasons.append(f"collapse={metrics.get('collapse'):.2f} on {metrics.get('top_pred')}")
            nxt["cap"] = 10.0
            nxt["soft_w"] = 0.5
        if metrics.get("lift", 0) < GOALS["lift_min"]:
            reasons.append(f"lift={metrics.get('lift'):+.3f}")
            nxt["hard_w"] = 1.2
            nxt["lr"] = 3e-4
        if weak:
            reasons.append("weak_classes=" + ",".join(f"{a}:{c:.2f}/n{n}" for a, c, n in weak[:5]))
            nxt["cap"] = 12.0
        # one-at-a-time escalation by iteration
        if it >= 3:
            nxt["epochs"] = 6
            reasons.append("raise epochs->6")
        if it >= 4:
            nxt["r"] = 32
            reasons.append("raise lora r->32")
        if it >= 5:
            nxt["hard_w"] = 1.5
            nxt["soft_w"] = 0.2
            reasons.append("harder CE, softer distill")
        if prev and prev.get("metrics"):
            d = metrics.get("val_top1", 0) - prev["metrics"].get("val_top1", 0)
            if d < 0.005 and it >= 2:
                reasons.append(f"plateau dTop1={d:+.3f}")
                nxt["lr"] = max(1e-4, nxt["lr"] * 0.7)
    r.update({"ok_model": bool(ok), "reasons": reasons, "next": nxt,
              "n_classes_pred": n_classes_pred, "weak": weak[:8]})
    return r


def write_loop_refl(entry: dict) -> None:
    REFL.mkdir(exist_ok=True)
    p = REFL / "loop_model.json"
    hist = []
    if p.is_file():
        try:
            hist = json.loads(p.read_text())
        except json.JSONDecodeError:
            hist = []
    if isinstance(hist, dict):
        hist = [hist]
    hist.append(entry)
    p.write_text(json.dumps(hist, indent=2))
    # also human-readable
    with (REFL / "loop_model.md").open("a") as f:
        f.write(f"\n## iter {entry['iter']} ok={entry.get('ok_model')}\n")
        f.write(f"- metrics: `{json.dumps(entry.get('metrics'))}`\n")
        f.write(f"- reasons: {entry.get('reasons')}\n")
        f.write(f"- next knobs: `{json.dumps(entry.get('next'))}`\n")


def train_once(knobs: dict, resume: bool) -> int:
    epochs = int(knobs.get("epochs", 4))
    # OneCycle needs total_steps to grow when extending epochs on a finished run.
    # train_lora_sf2 computes n_steps from args.epochs — extending epochs resumes and trains more.
    cmd = [
        PY, "train_lora_sf2.py",
        "--epochs", str(epochs),
        "--lr", str(knobs.get("lr", 2e-4)),
        "--r", str(knobs.get("r", 16)),
        "--ckpt-every", "100",
    ]
    if resume:
        cmd.append("--resume")
    else:
        cmd.append("--no-resume")
    os.environ["SF2_HARD_W"] = str(knobs.get("hard_w", 1.0))
    os.environ["SF2_SOFT_W"] = str(knobs.get("soft_w", 0.35))
    os.environ["SF2_PR_W"] = str(knobs.get("pr_w", 0.15))
    os.environ["SF2_WCAP"] = str(knobs.get("cap", 8.0))
    log(f"train knobs hard_w={os.environ['SF2_HARD_W']} soft_w={os.environ['SF2_SOFT_W']} "
        f"pr_w={os.environ['SF2_PR_W']} cap={os.environ['SF2_WCAP']} epochs={epochs}")
    return run(cmd, timeout=3600, stream_to=Path("/tmp/sf2_train_cur.log"))


def eval_once() -> tuple[int, dict | None]:
    rc = run([PY, "eval_lora_sf2.py"], timeout=900, stream_to=Path("/tmp/sf2_eval_cur.log"))
    return rc, load_eval()


def reflect_play(stage: str) -> dict:
    """Read decision log for collapse / damage."""
    p = ROOT / f"sf2_decisions_{stage}.jsonl"
    entry = {"stage": stage, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not p.is_file():
        entry.update({"ok_play": False, "reasons": ["no decision log"]})
        return entry
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    lora_c = Counter(r.get("lora") for r in rows if r.get("lora"))
    whys = Counter(r.get("why") for r in rows)
    res_p = Path(str(p) + ".result.json")
    res = json.loads(res_p.read_text()) if res_p.is_file() else {}
    # attack share among lora picks
    attack = {"light", "heavy", "fireball", "anti_air", "punish", "finish", "jump_in"}
    n_l = sum(lora_c.values()) or 1
    atk_share = sum(lora_c.get(a, 0) for a in attack) / n_l
    top_share = (max(lora_c.values()) / n_l) if lora_c else 1.0
    reasons = []
    ok = res.get("outcome") == "WIN" and top_share <= 0.65 and atk_share >= 0.25
    if res.get("outcome") != "WIN":
        reasons.append(f"outcome={res.get('outcome')} hp={res.get('hp')} ehp={res.get('ehp')}")
    if top_share > 0.65:
        reasons.append(f"lora collapse {top_share:.2f} on {lora_c.most_common(1)}")
    if atk_share < 0.25:
        reasons.append(f"attack_share={atk_share:.2f}")
    entry.update({
        "ok_play": bool(ok),
        "outcome": res.get("outcome"),
        "ticks": res.get("ticks"),
        "hp": res.get("hp"), "ehp": res.get("ehp"),
        "lora_hist": dict(lora_c),
        "top_why": whys.most_common(8),
        "atk_share": round(atk_share, 3),
        "top_share": round(top_share, 3),
        "reasons": reasons,
        "n": len(rows),
    })
    REFL.mkdir(exist_ok=True)
    with (REFL / "loop_play.md").open("a") as f:
        f.write(f"\n## {stage} {entry['ts']} ok={ok} outcome={res.get('outcome')}\n")
        f.write(f"- lora: {dict(lora_c)}\n- reasons: {reasons}\n")
    (REFL / "loop_play.json").write_text(json.dumps(entry, indent=2))
    return entry


def chain_done() -> list[str]:
    p = ROOT / "chain_status.json"
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text()).get("done") or []
    except json.JSONDecodeError:
        return []


def smoke(stage: str = "ken_ryu_4") -> int:
    return run([
        PY, "laya_sf2.py", "--state", stage, "--max-frames", "9000",
        "--lora", "--strategy", "aggressive",
    ], timeout=480)


def prepare_chain_resume() -> None:
    p = ROOT / "chain_status.json"
    st = json.loads(p.read_text()) if p.is_file() else {"done": [], "failed": [], "reflections": {}}
    st["failed"] = []
    st.setdefault("reflections", {})
    st.setdefault("active", None)
    p.write_text(json.dumps(st, indent=1))
    log(f"chain status: done={st.get('done')} active={st.get('active')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-stage", choices=["model", "play", "chain"], default="model")
    ap.add_argument("--max-model-iters", type=int, default=8)
    ap.add_argument("--smoke-stage", default="ken_ryu_4")
    ap.add_argument("--train-iters-per-visit", type=int, default=2,
                    help="max train+eval cycles before forcing a play smoke")
    args = ap.parse_args()

    log(f"=== loop start from={args.from_stage} ===")
    model_iter = 0
    prev_metrics = None
    # seed knobs from last reflection if any
    knobs = {"epochs": 4, "lr": 2e-4, "r": 16, "hard_w": 1.0, "soft_w": 0.35, "pr_w": 0.15, "cap": 8.0}
    refl_p = REFL / "loop_model.json"
    if refl_p.is_file():
        try:
            hist = json.loads(refl_p.read_text())
            if hist:
                knobs = hist[-1].get("next") or knobs
                prev_metrics = hist[-1].get("metrics")
                model_iter = int(hist[-1].get("iter", 0))
                log(f"resume loop at iter={model_iter} knobs={knobs}")
        except json.JSONDecodeError:
            pass

    # --- MODEL phase ---
    if args.from_stage == "model":
        while model_iter < args.max_model_iters:
            model_iter += 1
            log(f"--- MODEL iter {model_iter} knobs={knobs} ---")
            fresh = model_iter == 1 and not (OUT / "model_full.pt").is_file()
            rc = train_once(knobs, resume=not fresh)
            if rc not in (0,):
                # resume-able: if failed mid-way try once more with resume
                log(f"train rc={rc}; retry resume")
                rc = train_once(knobs, resume=True)
            if rc != 0:
                log(f"train failed rc={rc}")
                return rc
            rc, metrics = eval_once()
            if rc != 0:
                log(f"eval failed rc={rc}")
                return rc
            entry = reflect_model(model_iter, metrics, prev_metrics)
            write_loop_refl(entry)
            log(f"reflect ok={entry['ok_model']} reasons={entry.get('reasons')}")
            prev_metrics = metrics
            knobs = entry.get("next") or knobs
            if entry["ok_model"]:
                archive_run(f"v{model_iter}_good")
                log("MODEL GOALS MET -> play smoke")
                break
            # after a few iters, smoke anyway to see play signal
            if model_iter % args.train_iters_per_visit == 0:
                log("mid-loop play smoke")
                prc = smoke(args.smoke_stage)
                pe = reflect_play(args.smoke_stage)
                log(f"play smoke ok={pe.get('ok_play')} {pe.get('reasons')}")
                if pe.get("ok_play"):
                    archive_run(f"v{model_iter}_playok")
                    args.from_stage = "chain"
                    break
            # next knobs applied next train (fresh only if r/epochs changed? always resume if same arch)
            if knobs.get("r") != 16 and (OUT / "checkpoints" / "ckpt_last.pt").exists():
                # rank change → must restart weights
                archive_run(f"v{model_iter}_pre_rchange")
                shutil.rmtree(OUT / "checkpoints", ignore_errors=True)
                (OUT / "model_full.pt").unlink(missing_ok=True)
        else:
            log("max model iters without full goal; force smoke -> chain anyway")

    # --- PLAY phase ---
    if args.from_stage in ("model", "play"):
        for attempt in range(3):
            log(f"--- PLAY smoke attempt {attempt+1} ---")
            # prefer latest good archive; OUT already has latest train
            smoke(args.smoke_stage)
            pe = reflect_play(args.smoke_stage)
            log(f"play: {pe}")
            if pe.get("ok_play"):
                args.from_stage = "chain"
                break
            # change training from play signal
            knobs = {**(knobs or {})}
            knobs["epochs"] = int(knobs.get("epochs", 4)) + 2  # force more steps on resume
            knobs["lr"] = max(5e-5, float(knobs.get("lr", 2e-4)) * 0.8)
            if pe.get("top_share", 0) > 0.65:
                knobs["cap"] = 6.0
                knobs["soft_w"] = 0.5
                knobs["hard_w"] = 1.0
            if (pe.get("atk_share") or 1) < 0.25:
                knobs["hard_w"] = 1.3
                knobs["cap"] = 10.0
            # died while dealing damage → keep attacks, up block/finish via finish weight not easy;
            # prefer survival: slight soft up on defensive mass via more epochs
            if pe.get("hp", 1) is not None and int(pe.get("hp") or 0) <= 0 and int(pe.get("ehp") or 999) < 120:
                knobs["hard_w"] = 1.1
                knobs["soft_w"] = 0.45  # soften sharp argmax → more blocks from soft labels
            log(f"play-driven knobs={knobs}")
            rc = train_once(knobs, resume=True)
            if rc != 0:
                return rc
            rc, metrics = eval_once()
            if rc != 0:
                return rc
            model_iter += 1
            entry = reflect_model(model_iter, metrics, prev_metrics)
            write_loop_refl(entry)
            prev_metrics = metrics
            knobs = entry.get("next") or knobs
        else:
            log("play smoke never OK; continuing to chain with best model")

    # --- CHAIN phase ---
    if args.from_stage in ("model", "play", "chain"):
        prepare_chain_resume()
        done = chain_done()
        log(f"CHAIN start done={len(done)}/12 done={done}")
        log("launching sched_chain_sf2 ...")
        proc = subprocess.Popen(
            [PY, "sched_chain_sf2.py"],
            cwd=str(ROOT),
            stdout=open("/tmp/sf2_chain_stdout.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log(f"chain pid={proc.pid}")
        stall = 0
        last_done = len(done)
        while True:
            time.sleep(30)
            if proc.poll() is not None:
                log(f"chain exited rc={proc.returncode}")
                break
            done = chain_done()
            if len(done) >= 12:
                log("CHAIN 12/12 SUCCESS")
                break
            if len(done) != last_done:
                last_done = len(done)
                stall = 0
                log(f"progress {len(done)}/12")
            else:
                stall += 1
            # 15 min no progress → kill, reflect, retrain, relaunch
            if stall >= 30:
                log("chain stall 15m -> kill and loop back to model")
                proc.terminate()
                time.sleep(3)
                proc.kill()
                # play reflection from last unfinished stage
                missing = [s for s in STAGES if s not in done]
                if missing:
                    stage = missing[0]
                    pe = reflect_play(stage)
                    log(f"stall reflect {stage}: {pe.get('reasons')}")
                    knobs = {**knobs, "epochs": min(8, knobs.get("epochs", 4) + 1)}
                    rc = train_once(knobs, resume=True)
                    if rc != 0:
                        return rc
                    rc, metrics = eval_once()
                    if rc == 0 and metrics:
                        model_iter += 1
                        entry = reflect_model(model_iter, metrics, prev_metrics)
                        write_loop_refl(entry)
                        knobs = entry.get("next") or knobs
                prepare_chain_resume()
                proc = subprocess.Popen(
                    [PY, "sched_chain_sf2.py"],
                    cwd=str(ROOT),
                    stdout=open("/tmp/sf2_chain_stdout.log", "a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log(f"chain relaunched pid={proc.pid}")
                stall = 0
                last_done = len(chain_done())

        done = chain_done()
        if len(done) >= 12:
            log("DONE 12/12")
            return 0
        log(f"chain ended incomplete done={done}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
