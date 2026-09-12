#!/usr/bin/env python3
"""Download ONNX (and optional checkpoints) from every public MicroDuck HF model."""

from __future__ import annotations

import json
from pathlib import Path

import os

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError

# huggingface.co is often blocked here; the mirror works.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

ROOT = Path("/home/william/Workspace/e1901/microduck/hf_policies")
# Hardcoded: HF list_models search timed out from this machine.
REPOS = [
    "HannesVonEssen/microduck-stilts",
    "HannesVonEssen/microduck-running",
    "HannesVonEssen/microduck-swing",
    "HannesVonEssen/microduck-basketball",
    "fffiloni/microduck-polite-bow-b1d864",
    "RemiFabre/microduck-flamingo-cycle",
    "fffiloni/microduck-moonwalk-backward-55e6af",
    "RemiFabre/microduck-rough-walk-e",
    "RemiFabre/microduck-rough-walk-g",
    "Nupr-Haokun/microduck-step-up-head-brake",
    "Histochemichael/microduck-electric-slide-policy",
    "pollen-robotics/microduck-policies",
    "joanfox/microduck-happy-hop",
    "kyoungsim/microduck",
    "Brunocho/microduck-walk",
    "BurntBanksy/microduck-running-bootstrap",
    "q2p/microduck-beak-throw",
    "allen73/microduck-physical-ai",
    "Teethyfish/microduck-collision-flamingo-ii",
    "white100big/microduck-rl-policies",
    "Jasmin71/microduck-trained",
    "Nupr-Haokun/microduck-step-up",
    "youn485/microduck-policies",
    "hwihwalab/microduck-3d-bipedal-teleop",
    "Datawhale/Microduck-RL-4096x6000",
    "minjunglee27/microduck-models",
    "Datawhale/Microduck-BallKick-4096x6000",
    "kenpeter123/microduck_rl",
    "neil-jo/microduck-walk",
    "arabellako22/microduck-walk-seed42",
    "neil-jo/microduck-polite-bow",
    "XenderYang/microduck-rl-ydx-walk",
    "XenderYang/microduck-cloud-policies",
    "XenderYang/microduck-pose-gesture",
    "langli11/microduck-tricks",
    "tfrere/microduck-move-local-lookaround",
    "tfrere/microduck-move-a-duck-that-walks-leaning-fa",
    "tfrere/microduck-move-a-sneaky-duck-that-walks-in",
    "tfrere/microduck-move-a-forward-roll",
    "torchrl/microduck-skills",
    "tfrere/microduck-move-a-sneaky-duck-that-walks-in-2",
    "tfrere/microduck-move-a-low-crouch-walk",
    "zhoumiaosen/microduck-running",
    "zhoumiaosen/microduck-straight-running",
    "kenpeter123/microduck-rl",
    "tfrere/microduck-move-a-proud-parade-march-with-th",
    "koyogi-hf/microduck-policy-test",
]


def discover_repos(api: HfApi) -> list[str]:
    del api
    return list(REPOS)


def pick_files(files: list[str]) -> list[str]:
    wanted = []
    for name in files:
        lower = name.lower()
        if lower.endswith(".onnx"):
            wanted.append(name)
        elif name in {"config.json", "manifest.json", "README.md"}:
            wanted.append(name)
        elif lower.endswith("checkpoint.pt") or name.endswith("model.pt") or name.endswith("model_final.pt"):
            wanted.append(name)
    return wanted


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    repos = discover_repos(api)
    index: list[dict] = []
    for repo_id in repos:
        slug = repo_id.replace("/", "--")
        dest = ROOT / slug
        dest.mkdir(parents=True, exist_ok=True)
        files = None
        for attempt in range(4):
            try:
                files = list_repo_files(repo_id)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"list retry {attempt + 1} {repo_id}: {exc}", flush=True)
        if files is None:
            index.append({"repo": repo_id, "ok": False, "error": "list_repo_files failed"})
            continue
        onnx = [f for f in files if f.lower().endswith(".onnx")]
        picked = pick_files(files)
        saved = []
        for rel in picked:
            ok = False
            for attempt in range(4):
                try:
                    path = hf_hub_download(
                        repo_id=repo_id,
                        filename=rel,
                        local_dir=dest,
                    )
                    saved.append(rel)
                    print(f"OK {repo_id} {rel} -> {path}", flush=True)
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {attempt + 1} {repo_id} {rel}: {exc}", flush=True)
            if not ok:
                print(f"GIVE UP {repo_id} {rel}", flush=True)
        index.append(
            {
                "repo": repo_id,
                "ok": bool(onnx),
                "onnx": onnx,
                "saved": saved,
                "dir": str(dest),
            }
        )
    (ROOT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {ROOT / 'index.json'} repos={len(index)} with_onnx={sum(1 for r in index if r.get('ok'))}")


if __name__ == "__main__":
    main()
