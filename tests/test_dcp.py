import os
import tempfile
import torch
import pytest
from optimtensors.serde import safe_save_optimizer, safe_load_into_optimizer, safe_load_optimizer

# Check if PyTorch Distributed Checkpoint (DCP) is available
try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
    DCP_AVAILABLE = True
except ImportError:
    DCP_AVAILABLE = False


@pytest.mark.skipif(not DCP_AVAILABLE, reason="PyTorch Distributed Checkpoint (DCP) APIs not available")
def test_dcp_roundtrip():
    # 1. Initialize toy model and optimizer
    model = torch.nn.Linear(5, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 2. Populate optimizer state by running a forward/backward step
    x = torch.randn(2, 5)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    
    # Get original optimizer state dict values for validation
    original_state_dict = get_optimizer_state_dict(model, optimizer)
    
    # 3. Perform PyTorch Distributed Checkpoint (DCP) Save
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = dcp.FileSystemWriter(tmpdir)
        dcp.save(state_dict={"optimizer": original_state_dict}, storage_writer=writer)
        print(f"--> Saved checkpoint using PyTorch DCP to: {tmpdir}")
        
        # Verify files were created
        files = os.listdir(tmpdir)
        print(f"--> DCP files created: {files}")
        assert len(files) > 0, "No checkpoint files created by PyTorch DCP"
        
        # 4. Perform PyTorch Distributed Checkpoint (DCP) Load into a fresh model/optimizer
        fresh_model = torch.nn.Linear(5, 5)
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        
        # Construct template state dict structure first
        fresh_state_dict = get_optimizer_state_dict(fresh_model, fresh_optimizer)
        
        # Read from DCP checkpoint storage
        reader = dcp.FileSystemReader(tmpdir)
        dcp.load(state_dict={"optimizer": fresh_state_dict}, storage_reader=reader)
        
        # Set states back to the fresh optimizer
        set_optimizer_state_dict(fresh_model, fresh_optimizer, optim_state_dict=fresh_state_dict)
        
        # Validate that loaded state tensors match original
        loaded_state_dict = get_optimizer_state_dict(fresh_model, fresh_optimizer)
        orig_state = original_state_dict["state"]
        loaded_state = loaded_state_dict["state"]
        for param_name in orig_state:
            orig_param_state = orig_state[param_name]
            loaded_param_state = loaded_state[param_name]
            for state_name in orig_param_state:
                orig_val = orig_param_state[state_name]
                loaded_val = loaded_param_state[state_name]
                if isinstance(orig_val, torch.Tensor):
                    assert torch.allclose(orig_val, loaded_val), f"Mismatch in {param_name}.{state_name}"
                else:
                    assert orig_val == loaded_val, f"Mismatch in {param_name}.{state_name}"
                    
        print("--> SUCCESS: PyTorch Distributed Checkpoint (DCP) roundtrip completed with identical states!")


@pytest.mark.skipif(not DCP_AVAILABLE, reason="PyTorch Distributed Checkpoint (DCP) APIs not available")
def test_optimtensors_with_dcp_state_dict():
    # 1. Initialize toy model and optimizer
    model = torch.nn.Linear(5, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 2. Populate optimizer state
    x = torch.randn(2, 5)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    
    # Get uniform FQN-based state dictionary
    fqn_state_dict = get_optimizer_state_dict(model, optimizer)
    
    # 3. Serialize FQN-based state dictionary safely using optimtensors (no pickle)
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "optimizer.safetensors")
        safe_save_optimizer(fqn_state_dict, filepath)
        print(f"--> Saved FQN state dict using safe_save_optimizer to: {filepath}")
        
        # 4. Deserialize FQN-based state dictionary back into a fresh model/optimizer
        fresh_model = torch.nn.Linear(5, 5)
        fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
        
        # Load from safetensors file directly
        loaded_fqn_dict = safe_load_optimizer(filepath)
        
        # Set states back to the fresh optimizer
        set_optimizer_state_dict(fresh_model, fresh_optimizer, optim_state_dict=loaded_fqn_dict)
        
        # Validate that loaded state tensors match original
        loaded_state_dict = get_optimizer_state_dict(fresh_model, fresh_optimizer)
        orig_state = fqn_state_dict["state"]
        loaded_state = loaded_state_dict["state"]
        for param_name in orig_state:
            orig_param_state = orig_state[param_name]
            loaded_param_state = loaded_state[param_name]
            for state_name in orig_param_state:
                orig_val = orig_param_state[state_name]
                loaded_val = loaded_param_state[state_name]
                if isinstance(orig_val, torch.Tensor):
                    assert torch.allclose(orig_val, loaded_val), f"Mismatch in {param_name}.{state_name}"
                else:
                    assert orig_val == loaded_val, f"Mismatch in {param_name}.{state_name}"
                    
        print("--> SUCCESS: optimtensors serialized and loaded uniform FQN state dict successfully!")


if __name__ == "__main__":
    if DCP_AVAILABLE:
        test_dcp_roundtrip()
        test_optimtensors_with_dcp_state_dict()
    else:
        print("PyTorch Distributed Checkpoint (DCP) APIs not available. Skipping runtime tests.")
