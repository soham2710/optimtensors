import os
import tempfile
import pytest
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from optimtensors.serde import safe_save_optimizer, safe_load_optimizer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

def load_and_verify(path, expected_state_dict):
    try:
        loaded = safe_load_optimizer(path)
        for p_id in expected_state_dict["state"]:
            orig_p = expected_state_dict["state"][p_id]
            loaded_p = loaded["state"][p_id]
            for k in orig_p:
                v_orig = orig_p[k]
                v_load = loaded_p[k]
                if isinstance(v_orig, torch.Tensor):
                    v_sum = v_load.sum().item()
                    assert torch.equal(v_orig.cpu(), v_load)
                else:
                    assert v_orig == v_load
        return "SUCCESS"
    except Exception as e:
        return f"FAILED: {str(e)}"

def test_concurrent_reads():
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    
    state_dict = optimizer.state_dict()
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(state_dict, tmp_path)
        
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(load_and_verify, tmp_path, state_dict) for _ in range(num_threads)]
            results = [f.result() for f in futures]
            
        for r in results:
            assert r == "SUCCESS", f"Concurrent read failed with error: {r}"
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
