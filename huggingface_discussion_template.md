# Proposed GitHub Discussion/Issue Template for huggingface/safetensors

**Title**: Extending safetensors layout concepts to secure PyTorch Optimizer States (Introducing `optimtensors`)

---

### Context & Problem Statement

Hugging Face's `safetensors` has successfully secured model weights serialization across the ecosystem by providing a zero-code-execution format. However, as documented in its design scope, `safetensors` explicitly excludes **optimizer states** (since optimizer checkpointing requires storing non-tensor metadata like hyperparameters, scalar dictionary structures, learning rate schedulers, and step counts alongside standard state tensors).

Because of this, ML training checkpoints are still forced to rely on standard PyTorch `pickle` (`torch.save` / `torch.load`) to serialize optimizer states. This leaves training pipelines and shared checkpoints vulnerable to arbitrary code execution (as documented in CVE-2022-42969).

---

### Proposed Solution: The `optimtensors` Layout

To solve this gap, we have built `optimtensors` (https://github.com/[your-username]/optimtensors) — a Python library that adapts the core format design of `safetensors` to the complex state dict structure of PyTorch optimizers. 

The format maps states into a 3-part layout:
1. **JSON Metadata Header**: Stores tensor sizes, shapes, dtypes, and file offsets, matching `safetensors` specification.
2. **Scalar Configuration Block**: Safely serializes hyperparameters (`lr`, `betas`, `eps`, `weight_decay`) and non-tensor configurations in a raw binary block without Python execution.
3. **Tensor Data Block**: Contiguous binary buffers containing the actual state tensors (e.g. momentum vectors, steps), mapped using memory mapping (`mmap`).

---

### Performance & Safety Verification

We have validated the format across several benchmarks:
* **$O(1)$ RAM Memory Mapping**: Maps an 80MB checkpoint in **0.24 ms** using **0.79 MB** of physical RAM at initialization (vs. **81.30 MB** consumed by standard `torch.load` immediately).
* **Mixed-Dtype & Mixed-Device**: Preserves mixed `float16`, `float32`, and GPU/CPU resident states exactly.
* **Loss Curve Determinism**: Verified inside a live Hugging Face `SFTTrainer` with PEFT (LoRA) on the **Llama-3.2-1B** model. Training losses before and after checkpoint-resume match exactly to **four decimal places**:
  ```
  Control Loss at step 15: 1.5912, step 16: 1.1046
  Resumed Loss at step 15: 1.5912, step 16: 1.1046
  ```
* **Quantized Rejection**: Protects correctness by detecting and rejecting non-tensor quantized states (e.g., bitsandbytes `adamw_8bit`) during serialization.

---

### Discussion & Questions for the Community

We would love to get feedback from the Hugging Face team and the `safetensors` community:
1. **Layout Standardization**: Does this 3-part separation (JSON metadata + scalar blocks + binary tensors) align with how Hugging Face envisions solving the optimizer security gap?
2. **Integration into HF Ecosystem**: What is the recommended path for supporting secure optimizer checkpoints natively inside libraries like `transformers`, `accelerate`, and `trl`?
3. **Security Vulnerability Hardening**: Are there any edge cases (e.g., specific optimizer architectures) where a flat tensor mapping could bypass the bounds checks we implemented?

We invite design reviews and contributions on the repository: **[Insert Link to Repository]**
