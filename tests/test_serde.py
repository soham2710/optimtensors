import os
import tempfile
import pytest
import torch
import torch.nn as nn
from optimtensors.serde import safe_save_optimizer, safe_load_optimizer, safe_load_into_optimizer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def tensors_equal(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    if t1.shape != t2.shape or t1.dtype != t2.dtype:
        return False
    nan_mask1 = t1.isnan()
    nan_mask2 = t2.isnan()
    if not torch.equal(nan_mask1, nan_mask2):
        return False
    return torch.equal(t1.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), t2.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0))


@pytest.mark.parametrize("optim_class,kwargs", [
    (torch.optim.AdamW, {"lr": 0.001, "betas": (0.9, 0.995), "eps": 1e-7, "weight_decay": 0.01}),
    (torch.optim.Adam, {"lr": 0.001, "betas": (0.9, 0.99), "eps": 1e-8}),
    (torch.optim.SGD, {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.001}),
    (torch.optim.RMSprop, {"lr": 0.01, "momentum": 0.9, "alpha": 0.99}),
    (torch.optim.Adagrad, {"lr": 0.01, "lr_decay": 0.001}),
    (torch.optim.Adadelta, {"lr": 1.0, "rho": 0.9}),
    (torch.optim.LBFGS, {"lr": 1.0}),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_optimizer_roundtrip(optim_class, kwargs, dtype):
    # LBFGS only supports float32 and float64
    if optim_class == torch.optim.LBFGS and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("LBFGS does not support low precision dtypes")

    model = SimpleModel()
    if dtype in (torch.float16, torch.bfloat16):
        model = model.to(dtype)
        
    optimizer = optim_class(model.parameters(), **kwargs)
    
    # Run dummy steps to populate optimizer state
    for _ in range(3):
        x = torch.randn(4, 10, dtype=dtype)
        y = torch.randint(0, 2, (4,))
        
        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            return loss
            
        if optim_class == torch.optim.LBFGS:
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()

    # Get original state dict
    orig_state_dict = optimizer.state_dict()
    
    # Save optimizer state to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(orig_state_dict, tmp_path)
        
        # Load state dict back
        loaded_state_dict = safe_load_optimizer(tmp_path)
        
        # 1. Assert keys match
        assert set(orig_state_dict.keys()) == set(loaded_state_dict.keys())
        
        # 2. Check param_groups
        orig_groups = orig_state_dict["param_groups"]
        loaded_groups = loaded_state_dict["param_groups"]
        assert len(orig_groups) == len(loaded_groups)
        
        for g_orig, g_load in zip(orig_groups, loaded_groups):
            for k in g_orig:
                if k == "params":
                    assert g_orig[k] == g_load[k]
                elif isinstance(g_orig[k], (list, tuple)):
                    assert list(g_orig[k]) == list(g_load[k])
                else:
                    assert g_orig[k] == g_load[k]
                    
        # 3. Check parameter states
        orig_state = orig_state_dict["state"]
        loaded_state = loaded_state_dict["state"]
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
                    assert tensors_equal(val_orig, val_load)
                elif isinstance(val_orig, (list, tuple)) and len(val_orig) > 0 and any(isinstance(x, torch.Tensor) for x in val_orig):
                    assert isinstance(val_load, (list, tuple))
                    assert len(val_orig) == len(val_load)
                    for t_orig, t_load in zip(val_orig, val_load):
                        if t_orig is None:
                            assert t_load is None
                        else:
                            assert isinstance(t_load, torch.Tensor)
                            assert t_orig.shape == t_load.shape
                            assert t_orig.dtype == t_load.dtype
                            assert tensors_equal(t_orig, t_load)
                else:
                    assert val_orig == val_load

        # 4. Load back and verify it runs fine
        new_model = SimpleModel()
        if dtype in (torch.float16, torch.bfloat16):
            new_model = new_model.to(dtype)
            
        new_optimizer = optim_class(new_model.parameters(), **kwargs)
        new_optimizer.load_state_dict(loaded_state_dict)
        
        # Verify one more step
        x = torch.randn(4, 10, dtype=dtype)
        y = torch.randint(0, 2, (4,))
        if optim_class == torch.optim.LBFGS:
            new_optimizer.step(lambda: nn.CrossEntropyLoss()(new_model(x), y))
        else:
            new_optimizer.zero_grad()
            output = new_model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            new_optimizer.step()
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_safe_load_into_optimizer_validation():
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        safe_save_optimizer(optimizer.state_dict(), tmp_path)
        
        new_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        safe_load_into_optimizer(new_optimizer, tmp_path)
        assert new_optimizer.param_groups[0]["lr"] == 0.01
        
        class DifferentModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 6)
                self.fc2 = nn.Linear(6, 2)
            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))
                
        diff_model = DifferentModel()
        diff_optimizer = torch.optim.Adam(diff_model.parameters(), lr=0.01)
        
        with pytest.raises(ValueError) as excinfo:
            safe_load_into_optimizer(diff_optimizer, tmp_path)
        assert "Shape mismatch" in str(excinfo.value)
        
        diff_optimizer_groups = torch.optim.Adam([
            {"params": list(model.fc1.parameters()), "lr": 0.01},
            {"params": list(model.fc2.parameters()), "lr": 0.005}
        ])
        with pytest.raises(ValueError) as excinfo:
            safe_load_into_optimizer(diff_optimizer_groups, tmp_path)
        assert "Parameter group count mismatch" in str(excinfo.value)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_mixed_precision_roundtrip():
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    
    state_dict = optimizer.state_dict()
    
    # Manually inject fp16 and fp32 tensors into the state dict
    for param_id, param_state in state_dict["state"].items():
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"].to(torch.float32)
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"].to(torch.float16)
            
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(state_dict, tmp_path)
        loaded_state_dict = safe_load_optimizer(tmp_path)
        
        for param_id in state_dict["state"]:
            orig_p = state_dict["state"][param_id]
            loaded_p = loaded_state_dict["state"][param_id]
            assert loaded_p["exp_avg"].dtype == torch.float32
            assert loaded_p["exp_avg_sq"].dtype == torch.float16
            assert torch.equal(orig_p["exp_avg"], loaded_p["exp_avg"])
            assert torch.equal(orig_p["exp_avg_sq"], loaded_p["exp_avg_sq"])
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_gpu_roundtrip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    model = SimpleModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    x = torch.randn(4, 10).cuda()
    y = torch.randint(0, 2, (4,)).cuda()
    optimizer.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    
    orig_state_dict = optimizer.state_dict()
    
    has_cuda_tensor = False
    for param_id, state in orig_state_dict["state"].items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                if k in ("exp_avg", "exp_avg_sq"):
                    assert v.is_cuda
                    has_cuda_tensor = True
    assert has_cuda_tensor
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(orig_state_dict, tmp_path)
        
        loaded_state_dict = safe_load_optimizer(tmp_path)
        
        for param_id in orig_state_dict["state"]:
            orig_p = orig_state_dict["state"][param_id]
            loaded_p = loaded_state_dict["state"][param_id]
            for k in orig_p:
                if isinstance(orig_p[k], torch.Tensor):
                    assert not loaded_p[k].is_cuda
                    assert torch.equal(orig_p[k].cpu(), loaded_p[k])
                    
        new_model = SimpleModel().cuda()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        safe_load_into_optimizer(new_optimizer, tmp_path)
        
        for p in new_optimizer.state:
            for k, v in new_optimizer.state[p].items():
                if isinstance(v, torch.Tensor):
                    if k in ("exp_avg", "exp_avg_sq"):
                        assert v.is_cuda
                    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_adam_adamw_classification_regression():
    import json
    import struct
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
        # Test warning is raised and optimizer_type is "Adam" when auto-inferring
        with pytest.warns(UserWarning, match="Optimizer type inferred as 'Adam'"):
            safe_save_optimizer(state_dict, tmp_path)
            
        # Verify the saved header has "Adam", not "AdamW"
        with open(tmp_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
            assert header["__metadata__"]["optimizer_type"] == "Adam"
            
        # Test override source of truth
        safe_save_optimizer(state_dict, tmp_path, optimizer_type="AdamW")
        with open(tmp_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
            assert header["__metadata__"]["optimizer_type"] == "AdamW"
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_0dim_tensor_single_element_parameter_validation():
    """Verify that uninitialized loaders correctly bypass shape validation for 0-dim tensors on numel==1 parameters."""
    class SingleElementModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Bias has shape [1] (numel == 1)
            self.linear = nn.Linear(1, 1, bias=True)
            
        def forward(self, x):
            return self.linear(x)
            
    model = SingleElementModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Train 1 step to populate optimizer state (including 0-dim 'step' tensor)
    x = torch.randn(2, 1)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(optimizer.state_dict(), tmp_path, optimizer_type="AdamW")
        
        # Create a fresh optimizer (no state initialized)
        fresh_model = SingleElementModel()
        fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.001)
        
        # Load state; should not raise ValueError on shape check of 0-dim 'step' tensor
        safe_load_into_optimizer(fresh_optimizer, tmp_path)
        
        # Verify state was correctly loaded
        fresh_state = fresh_optimizer.state_dict()["state"]
        for p_id, state in fresh_state.items():
            assert "step" in state
            # Step tensor should be 0-dimensional
            assert state["step"].dim() == 0
            assert int(state["step"]) == 1
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

