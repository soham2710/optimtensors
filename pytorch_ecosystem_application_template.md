# PyTorch Ecosystem Working Group Listing Application Template

This template contains the proposed information for applying to list `optimtensors` in the official PyTorch Ecosystem Directory (https://pytorch.org/ecosystem/).

---

### Project Information

* **Project Name**: `optimtensors`
* **GitHub Repository**: `https://github.com/[your-username]/optimtensors`
* **License**: MIT
* **Primary Contact**: [Your Name/Email]
* **Category**: Developer Tools / Security / Performance Optimization

---

### Description

`optimtensors` is a secure, high-performance, zero-code-execution serialization format designed specifically for PyTorch optimizer states. 

By utilizing memory mapping (`mmap`) and a structured 3-part layout (JSON metadata header, raw scalar configuration bytes, and contiguous tensor blocks), `optimtensors` enables ML developers to safely checkpoint and resume large-scale model training without the security risk of Python code execution inherent in PyTorch's default `pickle`-based serialization.

---

### Value to the PyTorch Ecosystem

1. **Plugging a Critical Security Vulnerability (CVE-2022-42969)**: While `safetensors` has successfully secured model weights, it explicitly excludes optimizer states from its scope. Consequently, PyTorch users still use insecure `torch.save` / `torch.load` for training checkpoints. `optimtensors` plugs this security hole, providing a safe alternative for the training phase.
2. **$O(1)$ Memory Mapping**: Resolves memory spikes during checkpoint loading. Tensors are virtually mapped and paged into physical RAM only upon access.
3. **Loss Curve Identity**: Aligned framework integration tests on `SFTTrainer` with PEFT (LoRA) Llama-3.2-1B training demonstrate that training resume is 100% mathematically identical to uninterrupted runs (losses match to 4 decimal places before, at, and after the checkpoint resume step).

---

### PyTorch Integration Details

`optimtensors` is fully compatible with standard PyTorch `Optimizer` states. It implements two clean APIs that act as safe drop-in replacements for state serialization:

```python
from optimtensors.serde import safe_save_optimizer, safe_load_into_optimizer

# 1. Secure serialization of state dict
safe_save_optimizer(optimizer.state_dict(), "optimizer.safetensors")

# 2. Secure deserialization and in-place loading (casts types and devices automatically)
safe_load_into_optimizer(optimizer, "optimizer.safetensors")
```

---

### Supporting Evidence & Links

* **WALKTHROUGH & BENCHMARKS**: Benchmark results showing an 80MB checkpoint mapped in **0.24 ms** with **0.79 MB** of RAM overhead, along with BERT/GPT-2 scale statistics on GPU, are documented in the README.
* **TEST SUITE**: A comprehensive suite of 48 tests (covering memory leaks, mixed-precision round-trips, concurrent reads, adversarial fuzzing, and live framework resume).
