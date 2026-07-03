# optimtensors 🚀

A `safetensors`-style, zero-code-execution serialization format and drop-in API for PyTorch optimizer checkpoints.

This library closes the security hole left by PyTorch's default checkpointing pipeline, where model weights have largely transitioned to `safetensors` but optimizer state checkpoints (which safetensors explicitly scopes out) still rely on Python's highly vulnerable `pickle` serialization format.

**Compatibility**: Python `>= 3.8`, PyTorch `>= 2.0.0`. The optional DCP integration (`SecureFileSystemWriter`/`SecureFileSystemReader`) requires a recent PyTorch with modern `torch.distributed.checkpoint` internals; on older versions the core package still imports and `optimtensors.DCP_AVAILABLE` is `False`.

---

## 🔒 Security Model

Unlike traditional `pickle` or denylist-based sanitizers (e.g. `picklescan`) which continue to suffer from bypasses, `optimtensors` is **secure by design**:
* **No code execution**: The loader strictly parses JSON to read metadata and constructs PyTorch tensors using `torch.frombuffer` directly from the raw binary data.
* **No `eval()`, `exec()`, or `pickle`**: No slots exist in the file format to register or execute arbitrary code.
* **Closed Type System**: Enforces a strict closed set of safe primitive types (`int`, `float`, `bool`, `str`, `None`, and homogeneous lists of these). If any unauthorized object is encountered during serialization, `optimtensors` raises an error immediately rather than silently ignoring it or falling back to pickle.

---

## 📂 File Layout

The file layout mirrors `safetensors` and is optimized for memory-mapped, zero-copy loading:

```
[8 bytes]   header length (uint64, little-endian)
[N bytes]   JSON header (UTF-8, space-padded to 8-byte alignment)
[remaining] raw tensor bytes buffer (each tensor padded to 8-byte alignment)
```

### JSON Header Schema
The JSON header contains four top-level keys:
1. `__metadata__`: File format metadata (e.g., `format_version`, `optimizer_type`).
2. `__tensors__`: Dict mapping tensor names to their type, shape, and byte offsets within the raw buffer.
3. `__scalars__`: Dict containing primitive states (e.g. scalar step count, etc.).
4. `__config__`: Dict containing structured hyperparameter configuration (like `param_groups`).

---

## 💻 Usage

`optimtensors` is designed to be a drop-in replacement for PyTorch's native optimizer save/load calls.

### Saving Optimizer State

```python
from optimtensors import safe_save_optimizer

# Before (Unsafe pickle)
# torch.save(optimizer.state_dict(), "checkpoint.pt")

# After (Safe, secure-by-design)
safe_save_optimizer(optimizer.state_dict(), "checkpoint.safetensors")
```

### Loading Optimizer State

```python
from optimtensors import safe_load_optimizer

# Before (Unsafe pickle)
# optimizer.load_state_dict(torch.load("checkpoint.pt"))

# After (Safe, secure-by-design)
optimizer.load_state_dict(safe_load_optimizer("checkpoint.safetensors"))
```

### Loading and Validating (Recommended)

To load and validate that the checkpoint's tensor shapes, types, and counts match your optimizer instance (preventing silent bugs or shape mismatches):

```python
from optimtensors import safe_load_into_optimizer

# Loads state dict AND validates compatibility before applying it to the optimizer
safe_load_into_optimizer(optimizer, "checkpoint.safetensors")
```

---

## 📊 Benchmarks

Below are benchmark results comparing `optimtensors` with `torch.save`/`torch.load` (pickle) on a model with ~10M parameters:

| Metric | Pickle (torch.save/load) | optimtensors (Safe) | Speedup / Ratio |
| --- | --- | --- | --- |
| **Save Time (s)** | 0.070s | 0.080s | 1.14x |
| **Load Time (s)** | 0.028s | 0.00022s | **128x Speedup** |
| **File Size (MB)** | 80.04 MB | 80.04 MB | 1.00x |

*(Benchmarks run on CPU. Memory mapping ensures loading is extremely fast and light on RAM.)*

---

## 🚀 Hugging Face Transformers Integration

Secure your Hugging Face `Trainer` loops from pickle vulnerability with zero changes to your training logic.

### Installation
```bash
pip install optimtensors[transformers]
```

### Usage

1. **Callback (Save Side)**:
   Add `OptimTensorsCallback` to your trainer callbacks:
   ```python
   from optimtensors import OptimTensorsCallback

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset,
       # mode="dual" writes both, "strict" deletes optimizer.pt
       callbacks=[OptimTensorsCallback(mode="strict")]
   )
   ```

2. **Mixin (Load Side)**:
   Subclass your Trainer with `OptimTensorsTrainerMixin` to support pickle-free resumes:
   ```python
   from transformers import Trainer
   from optimtensors import OptimTensorsTrainerMixin

   class SafeTrainer(OptimTensorsTrainerMixin, Trainer):
       pass

   trainer = SafeTrainer(
       model=model,
       args=training_args,
       train_dataset=dataset,
       # ...
   )
   # Resumes seamlessly from .optimtensors
   trainer.train(resume_from_checkpoint="path/to/checkpoint")
   ```

3. **Manual Load Helper**:
   For custom workflows, load state dictionaries directly:
   ```python
   from optimtensors import load_trainer_optimizer
   
   load_trainer_optimizer("path/to/checkpoint", trainer.optimizer, device=args.device)
   ```

### v1.0 Out-of-Scope Design Focus
- **Scheduler & RNG States**: Currently, `scheduler.pt` and `rng_state.pth` remain pickled. Moving these to safe format is targeted for v1.1.
- **DeepSpeed / FSDP**: Sharded optimizer checkpoints in distributed systems are skipped by the callback and mixin, as they are handled natively by PyTorch Distributed Checkpoint (DCP).

---

## ⚠️ DCP Integration: Threat-Model Caveat

`SecureFileSystemWriter`/`SecureFileSystemReader` replace the **pickled `.metadata` file** of PyTorch Distributed Checkpoint with a validated JSON document. However, PyTorch's `FileSystemWriter` also serializes non-tensor items (BYTE_IO write items, e.g. `param_groups`) *inside the `.distcp` data files* via `torch.save`, and `FileSystemReader.read_data` deserializes them with `torch.load` (whether that call is `weights_only` depends on your PyTorch version). Using the secure classes therefore **reduces** the pickle attack surface of DCP checkpoints but does not yet eliminate it. Fully pickle-free DCP data files are planned for v2. The single-file `safe_save_optimizer`/`safe_load_optimizer` format contains no pickle anywhere.
