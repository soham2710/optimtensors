# Security Policy

`optimtensors` is a security-focused project: its purpose is to replace
pickle-based PyTorch optimizer checkpointing with a zero-code-execution
format. We take reports against that guarantee seriously.

## Reporting a Vulnerability

Please report suspected vulnerabilities **privately** via
[GitHub Security Advisories](https://github.com/soham2710/optimtensors/security/advisories/new)
rather than opening a public issue.

Include, where possible:

- A proof-of-concept checkpoint file or the code that generates it
- The `optimtensors`, `torch`, and Python versions used
- What the attacker-controlled input is and what behavior it triggers

You should receive an acknowledgement within 7 days. Please allow us a
90-day coordinated-disclosure window before publishing details.

## Threat Model

**In scope — these are bugs, report them:**

- Any path by which loading a malicious `.optimtensors` file executes code,
  imports a module, or calls `eval`/`exec`/`pickle`
- Buffer over-reads or out-of-bounds memory access via crafted headers
  (offsets, shapes, dtypes)
- Resource-exhaustion (memory/CPU) attacks via crafted headers that bypass
  the existing limits (50 MB header cap, tensor-list length cap,
  offset bounds- and overlap-checks)
- Type-confusion: smuggling a non-whitelisted type through
  `check_safe_structure` or the JSON header

**Out of scope / known limitations (v1):**

- `scheduler.pt` and `rng_state.pth` in Hugging Face Trainer checkpoints
  remain pickled (loaded with `torch.load(weights_only=True)`); migrating
  them to a safe format is planned for v1.1.
- The DCP integration (`SecureFileSystemWriter`/`SecureFileSystemReader`,
  on the `dcp-integration` branch) removes pickle from the `.metadata` file
  only. BYTE_IO items inside `.distcp` data files are still serialized by
  PyTorch via `torch.save`/`torch.load`. Fully pickle-free DCP data files
  are planned for v2.
- Denial of service via legitimately huge tensors (the format stores raw
  tensor bytes; a multi-GB file maps to a multi-GB tensor by design).
  Loading is `mmap`-based, so physical RAM is only consumed on access.
- Malicious *values* (e.g. NaN/Inf moments, poisoned optimizer state that
  degrades training). The format guarantees no code execution, not that
  the numbers are good for your model.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | ✅        |
