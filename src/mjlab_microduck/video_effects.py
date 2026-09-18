"""Minimal no-op stubs so render_checkpoint.py can import."""
from __future__ import annotations

class CrashEffects:
    def __init__(self, *args, **kwargs):
        pass
    def reset(self, *args, **kwargs):
        pass
    def update(self, *args, **kwargs):
        return None
    def draw(self, *args, **kwargs):
        pass

def configure_video_cfg(cfg, *args, **kwargs):
    return cfg

def fix_render_shadows(env_or_cfg, *args, **kwargs):
    return env_or_cfg
