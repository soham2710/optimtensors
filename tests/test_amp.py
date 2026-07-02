import os
import tempfile
import pytest
import torch
import torch.nn as nn
from optimtensors.serde import safe_save_optimizer, safe_load_optimizer, safe_load_into_optimizer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

def test_amp_fp16_fp32():
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    
    state_dict = optimizer.state_dict()
    
    for p_id, param_state in state_dict["state"].items():
        param_state["exp_avg"] = param_state["exp_avg"].to(torch.float32)
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"].to(torch.float16)
        
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(state_dict, tmp_path)
        loaded_state_dict = safe_load_optimizer(tmp_path)
        
        for p_id in state_dict["state"]:
            orig_p = state_dict["state"][p_id]
            loaded_p = loaded_state_dict["state"][p_id]
            
            assert loaded_p["exp_avg"].dtype == torch.float32
            assert loaded_p["exp_avg_sq"].dtype == torch.float16
            
            assert torch.equal(orig_p["exp_avg"], loaded_p["exp_avg"])
            assert torch.equal(orig_p["exp_avg_sq"], loaded_p["exp_avg_sq"])
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_amp_bf16_fp32_with_int16_collision():
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    
    state_dict = optimizer.state_dict()
    
    for p_id, param_state in state_dict["state"].items():
        param_state["exp_avg"] = param_state["exp_avg"].to(torch.float32)
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"].to(torch.bfloat16)
        param_state["real_int16_tensor"] = torch.tensor([1, 2, 3, 4], dtype=torch.int16)
        
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(state_dict, tmp_path)
        loaded_state_dict = safe_load_optimizer(tmp_path)
        
        for p_id in state_dict["state"]:
            orig_p = state_dict["state"][p_id]
            loaded_p = loaded_state_dict["state"][p_id]
            
            assert loaded_p["exp_avg"].dtype == torch.float32
            assert loaded_p["exp_avg_sq"].dtype == torch.bfloat16
            assert loaded_p["real_int16_tensor"].dtype == torch.int16
            
            assert torch.equal(orig_p["exp_avg"], loaded_p["exp_avg"])
            assert torch.equal(orig_p["exp_avg_sq"], loaded_p["exp_avg_sq"])
            assert torch.equal(orig_p["real_int16_tensor"], loaded_p["real_int16_tensor"])
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
