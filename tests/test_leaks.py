import os
import tempfile
import pytest
import torch
import torch.nn as nn
import resource
import gc
from optimtensors.serde import safe_save_optimizer, safe_load_optimizer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(500, 500)
    def forward(self, x):
        return self.fc(x)

def test_memory_leaks():
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(4, 500)
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(x).mean(), torch.tensor(0.0))
    loss.backward()
    optimizer.step()
    
    state_dict = optimizer.state_dict()
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(state_dict, tmp_path)
        
        for _ in range(10):
            loaded = safe_load_optimizer(tmp_path)
            for p_id in loaded["state"]:
                for k, v in loaded["state"][p_id].items():
                    if isinstance(v, torch.Tensor):
                        v.sum().item()
            del loaded
            gc.collect()
            
        gc.collect()
        rss_10 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        for _ in range(90):
            loaded = safe_load_optimizer(tmp_path)
            for p_id in loaded["state"]:
                for k, v in loaded["state"][p_id].items():
                    if isinstance(v, torch.Tensor):
                        v.sum().item()
            del loaded
            gc.collect()
            
        gc.collect()
        rss_100 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        growth_kb = rss_100 - rss_10
        print(f"Memory growth from cycle 10 to 100: {growth_kb} KB (rss_10: {rss_10} KB, rss_100: {rss_100} KB)")
        
        assert growth_kb < 5120, f"Memory leak detected! Peak RSS grew by {growth_kb} KB between cycle 10 and 100"
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
