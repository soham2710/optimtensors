import os
import tempfile
import time
import struct
import torch
import torch.nn as nn
from optimtensors import safe_save_optimizer, safe_load_optimizer, safe_load_into_optimizer

# 1. BERT-Base Architecture (110M parameters)
class BERTLayer(nn.Module):
    def __init__(self, hidden=768, intermediate=3072):
        super().__init__()
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.mlp_in = nn.Linear(hidden, intermediate)
        self.mlp_out = nn.Linear(intermediate, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        
    def forward(self, x):
        qkv = self.qkv(x)
        attn = self.out(qkv[:, :, :768])
        x = self.ln1(x + attn)
        mlp = self.mlp_out(torch.relu(self.mlp_in(x)))
        return self.ln2(x + mlp)

class BERTBaseSimulated(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(30522, 768)
        self.layers = nn.ModuleList([BERTLayer() for _ in range(12)])
        self.pooler = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, 2)
        
    def forward(self, x):
        h = self.embeddings(x)
        for layer in self.layers:
            h = layer(h)
        return self.classifier(self.pooler(h[:, 0]))

# 2. GPT-2 Architecture (124M parameters)
class GPT2Block(nn.Module):
    def __init__(self, hidden=768, intermediate=3072):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.Linear(hidden, hidden * 3)
        self.attn_out = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp_in = nn.Linear(hidden, intermediate)
        self.mlp_out = nn.Linear(intermediate, hidden)
        
    def forward(self, x):
        h = self.ln1(x)
        qkv = self.attn(h)
        attn = self.attn_out(qkv[:, :, :768])
        x = x + attn
        h = self.ln2(x)
        mlp = self.mlp_out(torch.relu(self.mlp_in(h)))
        return x + mlp

class GPT2Simulated(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50257, 768)
        self.wpe = nn.Embedding(1024, 768)
        self.blocks = nn.ModuleList([GPT2Block() for _ in range(12)])
        self.ln_f = nn.LayerNorm(768)
        self.lm_head = nn.Linear(768, 50257, bias=False)
        
    def forward(self, x):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        h = self.wte(x) + self.wpe(pos)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.lm_head(h)

def verify_state_dicts(orig, loaded):
    assert set(orig.keys()) == set(loaded.keys())
    assert len(orig["param_groups"]) == len(loaded["param_groups"])
    
    # Compare param groups
    for g_orig, g_load in zip(orig["param_groups"], loaded["param_groups"]):
        for k in g_orig:
            if k == "params":
                assert g_orig[k] == g_load[k]
            elif isinstance(g_orig[k], (list, tuple)):
                assert list(g_orig[k]) == list(g_load[k])
            else:
                assert g_orig[k] == g_load[k]
                
    orig_state = orig["state"]
    loaded_state = loaded["state"]
    assert set(orig_state.keys()) == set(loaded_state.keys())
    
    for param_id in orig_state:
        p_orig = orig_state[param_id]
        p_load = loaded_state[param_id]
        assert set(p_orig.keys()) == set(p_load.keys())
        
        for state_key in p_orig:
            val_orig = p_orig[state_key]
            val_load = p_load[state_key]
            
            if isinstance(val_orig, torch.Tensor):
                assert isinstance(val_load, torch.Tensor)
                assert val_orig.shape == val_load.shape
                assert val_orig.dtype == val_load.dtype
                # Force checking all values to trigger page faults
                if not torch.equal(val_orig.cpu(), val_load.cpu()):
                    raise ValueError(f"Tensor value mismatch for param {param_id} state {state_key}")
            elif isinstance(val_orig, (list, tuple)) and len(val_orig) > 0 and any(isinstance(x, torch.Tensor) for x in val_orig):
                assert isinstance(val_load, (list, tuple))
                assert len(val_orig) == len(val_load)
                for idx, (t_orig, t_load) in enumerate(zip(val_orig, val_load)):
                    if t_orig is None:
                        assert t_load is None
                    else:
                        assert isinstance(t_load, torch.Tensor)
                        assert t_orig.shape == t_load.shape
                        assert t_orig.dtype == t_load.dtype
                        if not torch.equal(t_orig.cpu(), t_load.cpu()):
                            raise ValueError(f"Tensor value mismatch for param {param_id} state {state_key}[{idx}]")
            else:
                assert val_orig == val_load

def run_scale_benchmark(name, model, optim_class, optim_kwargs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n==================================================")
    print(f"🚀 Running Scale Test for: {name} (on {device.upper()})")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {param_count / 1e6:.2f} Million")
    
    model = model.to(device)
    optimizer = optim_class(model.parameters(), **optim_kwargs)
    
    # Run a step to populate states
    if name == "BERT-Base":
        x = torch.randint(0, 30522, (4, 32), device=device)
        y = torch.randint(0, 2, (4,), device=device)
    else: # GPT-2
        x = torch.randint(0, 50257, (4, 32), device=device)
        y = torch.randint(0, 50257, (4, 32), device=device)
    
    output = model(x)
    if name == "BERT-Base":
        loss = nn.CrossEntropyLoss()(output, y)
    else:
        loss = nn.CrossEntropyLoss()(output.view(-1, 50257), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    state_dict = optimizer.state_dict()
        
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Save time
        t0 = time.perf_counter()
        safe_save_optimizer(state_dict, tmp_path, optimizer_type=optim_class.__name__)
        t_save = time.perf_counter() - t0
        
        # Move state dict to CPU to allow clearing CUDA cache before load validation
        cpu_state_dict = {}
        for k, v in state_dict.items():
            if k == "state":
                cpu_state_dict["state"] = {}
                for p_id, p_state in v.items():
                    cpu_state_dict["state"][p_id] = {}
                    for sk, sv in p_state.items():
                        if isinstance(sv, torch.Tensor):
                            cpu_state_dict["state"][p_id][sk] = sv.cpu()
                        elif isinstance(sv, (list, tuple)):
                            cpu_state_dict["state"][p_id][sk] = [x.cpu() if isinstance(x, torch.Tensor) else x for x in sv]
                        else:
                            cpu_state_dict["state"][p_id][sk] = sv
            else:
                cpu_state_dict[k] = v

        del optimizer
        del state_dict
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear OS page cache for the file to get cold load numbers first
        fd = os.open(tmp_path, os.O_RDWR)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
            
        # Load time (cold)
        t0 = time.perf_counter()
        loaded_dict_cold = safe_load_optimizer(tmp_path)
        t_load_cold = time.perf_counter() - t0
        
        # Verify time (cold) - forces actual disk read page faults
        t0 = time.perf_counter()
        verify_state_dicts(cpu_state_dict, loaded_dict_cold)
        t_verify_cold = time.perf_counter() - t0
        
        # Load time (warm) - now it's warm in page cache
        t0 = time.perf_counter()
        loaded_dict_warm = safe_load_optimizer(tmp_path)
        t_load_warm = time.perf_counter() - t0
        
        # Verify time (warm) - served from cache
        t0 = time.perf_counter()
        verify_state_dicts(cpu_state_dict, loaded_dict_warm)
        t_verify_warm = time.perf_counter() - t0
        
        # Validation load time
        t0 = time.perf_counter()
        new_optimizer = optim_class(model.parameters(), **optim_kwargs)
        safe_load_into_optimizer(new_optimizer, tmp_path)
        t_validation_load = time.perf_counter() - t0
        
        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        
        print(f"Save Time:             {t_save:.5f} s")
        print(f"Load Time (Warm):      {t_load_warm:.5f} s")
        print(f"Verify Time (Warm):    {t_verify_warm:.5f} s")
        print(f"Total Load+Verify (W): {t_load_warm + t_verify_warm:.5f} s")
        print(f"Load Time (Cold):      {t_load_cold:.5f} s")
        print(f"Verify Time (Cold):    {t_verify_cold:.5f} s")
        print(f"Total Load+Verify (C): {t_load_cold + t_verify_cold:.5f} s")
        print(f"Validation Load:       {t_validation_load:.5f} s")
        print(f"File Size:             {file_size_mb:.2f} MB")
        print(f"Status:                SUCCESS (Identical round-trip verified)")
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    # Test BERT-Base
    bert = BERTBaseSimulated()
    run_scale_benchmark("BERT-Base", bert, torch.optim.AdamW, {"lr": 1e-4, "weight_decay": 0.01})
    
    # Clean up BERT model from VRAM
    del bert
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test GPT-2
    gpt2 = GPT2Simulated()
    run_scale_benchmark("GPT-2", gpt2, torch.optim.AdamW, {"lr": 1e-4, "weight_decay": 0.01})
