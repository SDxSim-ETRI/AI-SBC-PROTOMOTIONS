# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Resolve the external ``maniflow`` package and load its checkpoints.

ManiFlow training currently lives in a separate checkout
(``ManiFlow_Policy/ManiFlow``). This module is the only place that knows
that: it makes ``maniflow`` importable and turns a training-workspace
checkpoint into an inference-ready policy without pulling in any of the
training-side machinery (workspace, dataset, wandb, env runners).

When the ManiFlow model code is eventually vendored into this repo (or pip
installed), resolution step 3 below picks it up automatically and no caller
has to change.

Runtime dependencies of the ManiFlow lowdim policy import chain:
``torch, einops, timm, termcolor, zarr, dill, hydra-core, omegaconf``.
"""

import logging
import os
import sys
from importlib import util as importlib_util
from pathlib import Path
from typing import Optional, Tuple

import torch

log = logging.getLogger(__name__)

# Conventional location of the ManiFlow checkout on dev machines. Used as the
# last-resort fallback; prefer the MANIFLOW_ROOT env var or an explicit arg.
DEFAULT_MANIFLOW_ROOT = Path.home() / "Projects" / "ManiFlow_Policy" / "ManiFlow"


def ensure_maniflow_importable(maniflow_root: Optional[str] = None) -> str:
    """Make the ``maniflow`` package importable and return its root directory.

    Resolution order:
      1. ``maniflow`` already imported in this process — reuse it.
      2. Explicit ``maniflow_root`` argument, then the ``MANIFLOW_ROOT``
         environment variable (both point at the directory *containing* the
         ``maniflow`` package, i.e. ``.../ManiFlow_Policy/ManiFlow``).
      3. Already installed/vendored package discoverable without path edits.
      4. Conventional sibling checkout (``DEFAULT_MANIFLOW_ROOT``).
    """
    # NOTE: maniflow is a namespace package (no __init__.py) — locate it via
    # __path__/submodule_search_locations, never via __file__/spec.origin.
    def _package_dir_valid(root: Path) -> bool:
        return (root / "maniflow" / "policy").is_dir()

    if "maniflow" in sys.modules:
        pkg_paths = list(sys.modules["maniflow"].__path__)
        found_root = str(Path(pkg_paths[0]).parent)
        if maniflow_root is not None and Path(maniflow_root).resolve() != Path(
            found_root
        ).resolve():
            log.warning(
                f"maniflow already imported from {found_root}; "
                f"ignoring requested root {maniflow_root}"
            )
        return found_root

    candidates = []
    if maniflow_root is not None:
        candidates.append(Path(maniflow_root))
    env_root = os.environ.get("MANIFLOW_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    for root in candidates:
        if _package_dir_valid(root):
            sys.path.insert(0, str(root))
            log.info(f"maniflow package resolved from: {root}")
            return str(root)
        log.warning(f"maniflow not found under requested root: {root}")

    spec = importlib_util.find_spec("maniflow")
    if spec is not None and spec.submodule_search_locations:
        found_root = str(Path(list(spec.submodule_search_locations)[0]).parent)
        log.info(f"maniflow package already importable from: {found_root}")
        return found_root

    if _package_dir_valid(DEFAULT_MANIFLOW_ROOT):
        sys.path.insert(0, str(DEFAULT_MANIFLOW_ROOT))
        log.info(f"maniflow package resolved from default: {DEFAULT_MANIFLOW_ROOT}")
        return str(DEFAULT_MANIFLOW_ROOT)

    raise ImportError(
        "Could not resolve the maniflow package. Pass maniflow_root=, set the "
        "MANIFLOW_ROOT environment variable to the directory containing the "
        "'maniflow' package (e.g. .../ManiFlow_Policy/ManiFlow), or install it."
    )


def discover_best_checkpoint(run_dir: str) -> str:
    """Pick the best topk checkpoint in a ManiFlow run directory.

    Chooses the ``epoch=*-val_loss=*.ckpt`` file with the lowest val_loss;
    falls back to ``latest.ckpt``. Note: ``latest.ckpt`` is periodically
    rewritten by an in-progress training run, so topk files are safer to read
    while training is still going.
    """
    import glob
    import re

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    candidates = []
    for p in glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt")):
        m = re.search(r"val_loss=([0-9.]+?)\.ckpt$", os.path.basename(p))
        if m:
            candidates.append((float(m.group(1)), p))
    if candidates:
        return min(candidates)[1]
    latest = os.path.join(ckpt_dir, "latest.ckpt")
    if os.path.isfile(latest):
        return latest
    raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")


def load_maniflow_policy(
    ckpt_path: str,
    device: str = "cuda:0",
    use_ema: Optional[bool] = None,
    maniflow_root: Optional[str] = None,
) -> Tuple[torch.nn.Module, "omegaconf.DictConfig", dict]:  # noqa: F821
    """Load a ManiFlow workspace checkpoint into an inference-ready policy.

    Bypasses the training workspace entirely: instantiates ``cfg.policy`` via
    hydra and loads the (EMA) policy state dict, which includes the fitted
    ``LinearNormalizer``. The checkpoint format is the one written by
    ``maniflow.workspace.base_workspace.BaseWorkspace.save_checkpoint``:
    ``{'cfg', 'state_dicts': {'model', 'ema_model', 'optimizer'}, 'pickles'}``.

    Args:
        ckpt_path: Path to the ``.ckpt`` file.
        device: Torch device for the policy.
        use_ema: Load the EMA weights. Default (None) follows
            ``cfg.training.use_ema``, matching what evaluation uses.
        maniflow_root: Optional override for the maniflow package location.

    Returns:
        (policy, cfg, info) — policy is in eval mode on ``device``; info holds
        checkpoint metadata (epoch, global_step, state key used).
    """
    ensure_maniflow_importable(maniflow_root)

    import dill
    import hydra
    from omegaconf import OmegaConf

    # ManiFlow configs use a custom ${eval:...} resolver (e.g. dataset padding).
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    ckpt_path = os.path.expanduser(str(ckpt_path))
    with open(ckpt_path, "rb") as f:
        payload = torch.load(
            f, pickle_module=dill, map_location="cpu", weights_only=False
        )
    cfg = payload["cfg"]

    if use_ema is None:
        use_ema = bool(cfg.training.use_ema)
    state_key = "ema_model" if use_ema else "model"
    if state_key not in payload["state_dicts"]:
        raise KeyError(
            f"'{state_key}' not in checkpoint state_dicts "
            f"(available: {list(payload['state_dicts'].keys())})"
        )

    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][state_key])
    policy.to(torch.device(device))
    policy.eval()

    normalizer_keys = list(policy.normalizer.params_dict.keys())
    if "action" not in normalizer_keys:
        raise RuntimeError(
            f"Loaded normalizer has no 'action' entry (got {normalizer_keys}); "
            "checkpoint does not look like a fitted ManiFlow policy."
        )

    pickles = payload.get("pickles", {})
    info = {
        "ckpt_path": str(ckpt_path),
        "state_key": state_key,
        "epoch": dill.loads(pickles["epoch"]) if "epoch" in pickles else None,
        "global_step": (
            dill.loads(pickles["global_step"]) if "global_step" in pickles else None
        ),
        "normalizer_keys": normalizer_keys,
    }
    log.info(
        f"Loaded ManiFlow policy from {ckpt_path} "
        f"(epoch={info['epoch']}, weights={state_key})"
    )
    return policy, cfg, info
