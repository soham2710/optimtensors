import os
import time
import tempfile
import torch
import torch.nn as nn
import resource
import gc
from optimtensors.serde import safe_save_optimizer, infer_optimizer_type

# 1. Models and definitions
class SimpleBenchmarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc2(torch.relu(self.fc(x)))

def get_bert_base_state():
    # Mimics BERT-base-sized state (110M parameters)
    # We populate the optimizer states manually to have identical memory/parameter layout
    from transformers import BertConfig, BertForSequenceClassification
    config = BertConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForSequenceClassification(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Mock a training step to populate states on active device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x = torch.randint(0, 30522, (2, 32), device=device)
    optimizer.zero_grad()
    out = model(x).logits
    loss = out.sum()
    loss.backward()
    optimizer.step()
    return optimizer.state_dict()

# 2. Section 8a: GPU-sync vs. Disk-write breakdown
def run_sync_vs_write_breakdown():
    print("Running GPU-sync vs. Disk-write breakdown for BERT-Base size...")
    state_dict = get_bert_base_state()
    
    # Extract tensors to write
    tensors_to_write = []
    state = state_dict.get("state", {})
    for param_id, param_state in state.items():
        for k, v in param_state.items():
            if isinstance(v, torch.Tensor):
                tensors_to_write.append(v)
            elif isinstance(v, (list, tuple)) and len(v) > 0 and any(isinstance(x, torch.Tensor) for x in v):
                for t in v:
                    if t is not None:
                        tensors_to_write.append(t)
                        
    # Measure GPU to CPU transfer (GPU synchronization)
    t0 = time.perf_counter()
    cpu_tensors = []
    for t in tensors_to_write:
        cpu_tensors.append(t.detach().cpu().contiguous())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gpu_sync = time.perf_counter() - t0
    
    # Measure Disk Write
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        t0 = time.perf_counter()
        safe_save_optimizer(state_dict, tmp_path, optimizer_type="AdamW")
        t_total_save = time.perf_counter() - t0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    t_disk_write = t_total_save - t_gpu_sync
    print(f"GPU Sync Time: {t_gpu_sync:.5f} s")
    print(f"Disk Write Time: {t_disk_write:.5f} s (Total Save: {t_total_save:.5f} s)")
    return t_gpu_sync, t_disk_write, t_total_save

# 3. Section 8b/c: Per-step training overhead & Pickle comparison
def run_fine_tuning_loop(mode, checkpoint_interval):
    # Runs 201 steps of a mock fine-tuning loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleBenchmarkModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    
    step_times = []
    
    for step in range(201):
        t0 = time.perf_counter()
        
        optimizer.zero_grad()
        out = model(x)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        optimizer.step()
        
        # Checkpointing
        if checkpoint_interval > 0 and step > 0 and step % checkpoint_interval == 0:
            state_dict = optimizer.state_dict()
            with tempfile.NamedTemporaryFile(suffix=".pt" if mode == "pickle" else ".safetensors", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                if mode == "safe":
                    safe_save_optimizer(state_dict, tmp_path, optimizer_type="AdamW")
                elif mode == "pickle":
                    torch.save(state_dict, tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        t_step = time.perf_counter() - t0
        # Ignore first 10 steps as warmup
        if step >= 10:
            step_times.append(t_step)
            
    avg_step_time = sum(step_times) / len(step_times)
    return avg_step_time

# 4. Section 8d: Save-path memory tracking (RSS over 100 saves)
def run_save_memory_tracking():
    print("Running save-path RSS memory tracking over 100 saves...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleBenchmarkModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    state_dict = optimizer.state_dict()
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Warmup
        for _ in range(10):
            safe_save_optimizer(state_dict, tmp_path, optimizer_type="AdamW")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        gc.collect()
        rss_10 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Save 90 more times
        for _ in range(90):
            safe_save_optimizer(state_dict, tmp_path, optimizer_type="AdamW")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        gc.collect()
        rss_100 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        growth_kb = rss_100 - rss_10
        print(f"Save memory growth cycle 10 to 100: {growth_kb} KB")
        return growth_kb
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    # 8a. Breakdown
    t_gpu_sync, t_disk_write, t_total_save = run_sync_vs_write_breakdown()
    
    # 8b/c. Training loop overheads
    print("Benchmarking training loop overheads...")
    avg_no_cp = run_fine_tuning_loop("none", 0)
    
    # Safe saves (optimtensors)
    avg_safe_50 = run_fine_tuning_loop("safe", 50)
    avg_safe_200 = run_fine_tuning_loop("safe", 200)
    
    # Pickle saves (torch.save)
    avg_pt_50 = run_fine_tuning_loop("pickle", 50)
    avg_pt_200 = run_fine_tuning_loop("pickle", 200)
    
    # Slowdowns
    slowdown_safe_50 = (avg_safe_50 - avg_no_cp) / avg_no_cp * 100
    slowdown_safe_200 = (avg_safe_200 - avg_no_cp) / avg_no_cp * 100
    slowdown_pt_50 = (avg_pt_50 - avg_no_cp) / avg_no_cp * 100
    slowdown_pt_200 = (avg_pt_200 - avg_no_cp) / avg_no_cp * 100
    
    # 8d. Save memory tracking
    growth_kb = run_save_memory_tracking()
    
    # Append results to test_results.md
    with open("test_results.md", "a") as f:
        f.write("\n## Section 8: In-loop Training Performance Results\n\n")
        f.write("### 8a. GPU-sync vs. Disk-write Breakdown (BERT-Base scale)\n")
        f.write(f"- **GPU Sync (detach + CPU move)**: {t_gpu_sync:.5f} s\n")
        f.write(f"- **Disk Write**: {t_disk_write:.5f} s\n")
        f.write(f"- **Total Save**: {t_total_save:.5f} s\n\n")
        
        f.write("### 8b/c. Fine-Tuning Step Time & Slowdown Matrix\n")
        f.write("| Config | Average Step Time | Slowdown (%) |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| Baseline (No Checkpoints) | {avg_no_cp:.5f} s | 0.00% |\n")
        f.write(f"| optimtensors (Safe) - N=50 | {avg_safe_50:.5f} s | {slowdown_safe_50:.2f}% |\n")
        f.write(f"| optimtensors (Safe) - N=200 | {avg_safe_200:.5f} s | {slowdown_safe_200:.2f}% |\n")
        f.write(f"| Pickle (`torch.save`) - N=50 | {avg_pt_50:.5f} s | {slowdown_pt_50:.2f}% |\n")
        f.write(f"| Pickle (`torch.save`) - N=200 | {avg_pt_200:.5f} s | {slowdown_pt_200:.2f}% |\n\n")
        
        f.write("### 8d. Save-Path RSS Memory Leak Check\n")
        f.write(f"- **Peak RSS Growth (cycle 10 to 100)**: {growth_kb} KB\n")
        
    print("Benchmarks complete and results written to test_results.md!")

if __name__ == "__main__":
    main()
