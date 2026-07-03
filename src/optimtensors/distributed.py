"""
Distributed / sharded optimizer-state helpers for optimtensors.

Scope — read this first
-----------------------
This module is the pragmatic v1 of "DCP support". It provides two working
strategies:

1. **FSDP full-state save on rank 0** (``save_fsdp_full_optimizer`` /
   ``load_fsdp_full_optimizer``): gather the sharded optimizer state into a
   single full state dict on rank 0 via FSDP's official APIs, then serialize
   it with optimtensors. Simple, correct, and covers the common "I train with
   FSDP and want one safe checkpoint file" case. The tradeoff is the same as
   FSDP's own FULL_STATE_DICT path: rank 0 must hold the entire optimizer
   state in host RAM during save.

2. **Per-rank sharded save** (``save_sharded_optimizer`` /
   ``load_sharded_optimizer``): every rank writes its local shard as
   ``optimizer-rank{r}-of-{w}.optimtensors``. No gather, no rank-0 memory
   spike. Constraint: resume requires the same world size and the same
   sharding, exactly like torch's legacy per-rank checkpointing.

What this module is **not**: a ``torch.distributed.checkpoint`` (DCP)
``StorageWriter``/``StorageReader`` implementation. Real DCP integration —
resharding across world sizes, deduplicated planning, async saves — is a
genuinely separate project and is the v2 milestone. Building it as a DCP
storage backend is the right long-term design (DCP handles planning and
resharding; optimtensors would only handle safe bytes-on-disk), and the
format work here carries over directly. It is deliberately not faked here.

The FQN problem (why the sidecar map exists)
--------------------------------------------
optimtensors v1 encodes tensor locations as ``state.{param_id}.{state_key}``
and parses with ``split(".", 2)``. Trainer-style optimizer state uses integer
param ids, so that's unambiguous. FSDP's ``optim_state_dict``, however, keys
state by fully-qualified parameter names — ``model.layers.0.weight`` — which
contain dots and would corrupt key parsing.

v1 workaround, implemented here: before saving, state keys are re-mapped to
integer indices (sorted FQN order, deterministic), and the index→FQN map is
written as a plain-JSON sidecar (``<file>.fqnmap.json``). On load, keys are
mapped back. The sidecar is pure JSON — no pickle, and it is validated with
the same closed-type discipline as the main format.

Format v2 should fold this map into ``__config__`` so the file is
self-contained; tracked as a known limitation.
"""

from __future__ import annotations

import json
import logging
import os
import re

import torch

from optimtensors.serde import safe_save_optimizer, safe_load_optimizer
from optimtensors.type_check import check_safe_structure

logger = logging.getLogger(__name__)

FQNMAP_SUFFIX = ".fqnmap.json"
_SHARD_RE = re.compile(r"^optimizer-rank(\d+)-of-(\d+)\.optimtensors$")


# ---------------------------------------------------------------------------
# FQN <-> integer index re-keying
# ---------------------------------------------------------------------------

def _rekey_state_to_indices(state_dict: dict) -> tuple[dict, dict[int, str]]:
    """
    Replace string (FQN) state keys with deterministic integer indices.

    Also rewrites ``param_groups[i]["params"]`` if those entries are FQNs
    (FSDP's optim_state_dict uses FQNs there too). Integer-keyed state dicts
    pass through untouched with an empty map.
    """
    state = state_dict.get("state", {})
    str_keys = [k for k in state.keys() if isinstance(k, str)]
    if not str_keys:
        return state_dict, {}

    fqns = sorted(state.keys(), key=str)
    fqn_to_idx = {fqn: i for i, fqn in enumerate(fqns)}
    idx_to_fqn = {i: fqn for fqn, i in fqn_to_idx.items()}

    new_state = {fqn_to_idx[k]: v for k, v in state.items()}

    new_groups = []
    for group in state_dict.get("param_groups", []):
        g = dict(group)
        params = g.get("params", [])
        g["params"] = [
            fqn_to_idx.get(p, p) if isinstance(p, str) else p for p in params
        ]
        new_groups.append(g)

    return {"state": new_state, "param_groups": new_groups}, idx_to_fqn


def _rekey_state_to_fqns(state_dict: dict, idx_to_fqn: dict[int, str]) -> dict:
    if not idx_to_fqn:
        return state_dict
    state = state_dict.get("state", {})
    new_state = {}
    for k, v in state.items():
        idx = int(k)
        if idx not in idx_to_fqn:
            raise ValueError(
                f"State index {idx} has no entry in the FQN map — "
                f"checkpoint and sidecar map are inconsistent."
            )
        new_state[idx_to_fqn[idx]] = v

    new_groups = []
    for group in state_dict.get("param_groups", []):
        g = dict(group)
        g["params"] = [
            idx_to_fqn.get(int(p), p) if isinstance(p, int) else p
            for p in g.get("params", [])
        ]
        new_groups.append(g)
    return {"state": new_state, "param_groups": new_groups}


def _write_fqn_map(path: str, idx_to_fqn: dict[int, str]) -> None:
    sidecar = path + FQNMAP_SUFFIX
    payload = {str(i): fqn for i, fqn in idx_to_fqn.items()}
    tmp = sidecar + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "index_to_fqn": payload}, f)
    os.replace(tmp, sidecar)


def _read_fqn_map(path: str) -> dict[int, str]:
    sidecar = path + FQNMAP_SUFFIX
    if not os.path.isfile(sidecar):
        return {}
    with open(sidecar, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Same closed-type discipline as the main format: reject anything that
    # isn't plain JSON scalars/containers, and enforce the exact shape.
    check_safe_structure(raw, "fqnmap")
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError(f"Unrecognized FQN map format in {sidecar}")
    mapping = raw.get("index_to_fqn", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"Malformed index_to_fqn in {sidecar}")
    out: dict[int, str] = {}
    for k, v in mapping.items():
        if not (isinstance(k, str) and k.isdigit() and isinstance(v, str)):
            raise ValueError(f"Malformed FQN map entry {k!r}: {v!r} in {sidecar}")
        out[int(k)] = v
    return out


# ---------------------------------------------------------------------------
# Public: generic save/load that tolerate FQN-keyed state dicts
# ---------------------------------------------------------------------------

def save_optimizer_state_dict(state_dict: dict, path: str, **kwargs) -> None:
    """
    Like safe_save_optimizer, but accepts FQN(string)-keyed state dicts by
    re-keying to integer indices and writing a JSON sidecar map.
    """
    rekeyed, idx_to_fqn = _rekey_state_to_indices(state_dict)
    safe_save_optimizer(rekeyed, path, **kwargs)
    if idx_to_fqn:
        _write_fqn_map(path, idx_to_fqn)


def load_optimizer_state_dict(path: str) -> dict:
    """
    Like safe_load_optimizer, but restores FQN keys if a sidecar map exists.
    """
    state_dict = safe_load_optimizer(path)
    idx_to_fqn = _read_fqn_map(path)
    return _rekey_state_to_fqns(state_dict, idx_to_fqn)


# ---------------------------------------------------------------------------
# Strategy 1: FSDP full-state on rank 0
# ---------------------------------------------------------------------------

def save_fsdp_full_optimizer(model, optimizer, path: str, **kwargs) -> None:
    """
    Gather FSDP-sharded optimizer state to rank 0 and save it there.

    Uses the official ``FSDP.optim_state_dict`` API (torch >= 2.0) with the
    module's configured state-dict settings; call
    ``FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT, ...)``
    beforehand if you need to control offload/rank0_only behavior.

    All ranks must call this (the gather is collective); only rank 0 writes.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    full_osd = FSDP.optim_state_dict(model, optimizer)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        save_optimizer_state_dict(full_osd, path, **kwargs)
        logger.info("optimtensors: saved gathered FSDP optimizer state to %s", path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_fsdp_full_optimizer(model, optimizer, path: str) -> None:
    """
    Load a full optimizer state dict saved by save_fsdp_full_optimizer and
    re-shard it into the live FSDP optimizer.

    All ranks must call this. Rank 0 (well, every rank here — the file is
    read locally on each rank for simplicity; a broadcast-from-rank-0
    optimization is a straightforward follow-up) reads the file, then
    ``FSDP.optim_state_dict_to_load`` handles the re-sharding.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    full_osd = load_optimizer_state_dict(path)
    sharded = FSDP.optim_state_dict_to_load(
        model=model, optim=optimizer, optim_state_dict=full_osd
    )
    optimizer.load_state_dict(sharded)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# Strategy 2: per-rank sharded save (no gather)
# ---------------------------------------------------------------------------

def _shard_name(rank: int, world_size: int) -> str:
    return f"optimizer-rank{rank}-of-{world_size}.optimtensors"


def save_sharded_optimizer(optimizer, directory: str, **kwargs) -> None:
    """
    Every rank writes its local optimizer shard. No gather, no rank-0 memory
    spike. Resume requires identical world size and sharding.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "save_sharded_optimizer requires an initialized process group; "
            "for single-process training use safe_save_optimizer directly."
        )
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, _shard_name(rank, world))
    save_optimizer_state_dict(optimizer.state_dict(), path, **kwargs)
    torch.distributed.barrier()
    if rank == 0:
        logger.info(
            "optimtensors: wrote %d optimizer shards to %s", world, directory
        )


def load_sharded_optimizer(optimizer, directory: str) -> None:
    """
    Load this rank's shard written by save_sharded_optimizer, after checking
    that the on-disk world size matches the current one (fail loudly rather
    than resume wrongly).
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "load_sharded_optimizer requires an initialized process group."
        )
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()

    on_disk = [
        m for m in (_SHARD_RE.match(f) for f in os.listdir(directory)) if m
    ]
    if not on_disk:
        raise FileNotFoundError(f"No optimizer shards found in {directory}")
    disk_world = int(on_disk[0].group(2))
    if disk_world != world:
        raise ValueError(
            f"World size mismatch: checkpoint in {directory} was written with "
            f"{disk_world} ranks, current run has {world}. Per-rank sharded "
            f"checkpoints cannot be resharded — use the FSDP full-state path "
            f"(or, in v2, the DCP backend) for elastic resume."
        )

    path = os.path.join(directory, _shard_name(rank, world))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing shard for rank {rank}: {path}")
    state_dict = load_optimizer_state_dict(path)
    optimizer.load_state_dict(state_dict)
    torch.distributed.barrier()
