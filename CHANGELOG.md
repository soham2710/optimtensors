# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-07-04

### Fixed
- Tensor bytes are now written from `data_ptr()` instead of
  `untyped_storage().data_ptr()`, fixing wrong bytes being serialized for
  contiguous views with a nonzero storage offset (e.g. `x[1:]`).
- `safe_save_optimizer` writes atomically (temp file + `os.replace`), so an
  interrupted save can no longer leave a corrupt checkpoint.
- Added `from __future__ import annotations` so the package imports on
  Python 3.8/3.9.
- FQN re-keying (`optimtensors.distributed`) now rejects state dicts that
  mix string and integer parameter keys instead of silently corrupting
  `param_groups["params"]` references.

### Security (hardened untrusted-file load path)
- Top-level header values must be JSON objects.
- Scalar entries must be dicts carrying `type` and `value`.
- Tensor shapes must be lists of non-negative integers.
- `data_offsets` must be well-formed two-element integer lists, and
  overlapping tensor regions are rejected.
- `tensor_list` placeholder length is capped at the number of tensors
  declared in the header (closes a memory-DoS vector).
- Clean `ValueError`s (instead of raw `IndexError`/`TypeError`/`KeyError`)
  for out-of-range tensor-list indices, unknown parameter ids, and
  non-integer state keys in `safe_load_into_optimizer`.

### Changed
- Requires `torch >= 2.0` (the code uses `untyped_storage()`); dropped the
  unused `safetensors` dependency.
- License metadata unified to Apache-2.0 across distributions.

## [1.0.0] - 2026-07-03

### Added
- Initial release: zero-code-execution serialization format for PyTorch
  optimizer state dicts (`safe_save_optimizer`, `safe_load_optimizer`,
  `safe_load_into_optimizer`).
- `mmap`-based loading with O(1) physical RAM overhead.
- Closed-type validation of all non-tensor state (`check_safe_structure`).
- bfloat16 support via int16 reinterpretation.
- Hugging Face Trainer integration (`OptimTensorsCallback`,
  `OptimTensorsTrainerMixin`, `load_trainer_optimizer`).
- FSDP helpers: rank-0 full-state and per-rank sharded save/load with a
  JSON sidecar FQN map (`optimtensors.distributed`).
