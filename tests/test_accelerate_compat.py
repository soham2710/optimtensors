import os
import random
import struct
import json
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from optimtensors import safe_save_state, safe_load_state, safe_save_optimizer, safe_load_optimizer


def assert_structures_equal(val1, val2):
    """Recursively checks that two nested structures (with tensors and arrays) are equal."""
    if type(val1) != type(val2):
        raise AssertionError(f"Type mismatch: {type(val1)} != {type(val2)}")

    if isinstance(val1, dict):
        assert set(val1.keys()) == set(val2.keys())
        for k in val1:
            assert_structures_equal(val1[k], val2[k])
    elif isinstance(val1, (list, tuple)):
        assert len(val1) == len(val2)
        for x, y in zip(val1, val2):
            assert_structures_equal(x, y)
    elif type(val1).__name__ == "Tensor":
        assert torch.equal(val1, val2), f"Tensor mismatch: {val1} vs {val2}"
    elif isinstance(val1, np.ndarray):
        assert np.array_equal(val1, val2), f"ndarray mismatch: {val1} vs {val2}"
    else:
        assert val1 == val2, f"Value mismatch: {val1} vs {val2}"


def test_scheduler_state_roundtrip():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Do a step
    x = torch.randn(5, 10)
    y = torch.randn(5, 2)
    loss = (model(x) - y).pow(2).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    orig_state = scheduler.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        safe_save_state(orig_state, tmp_path)
        loaded_state = safe_load_state(tmp_path)
        
        assert_structures_equal(orig_state, loaded_state)
        # Assert type identity
        assert type(loaded_state) is dict
        assert type(loaded_state["base_lrs"]) is list
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_rng_states_roundtrip():
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    
    cuda_rng = None
    if torch.cuda.is_available():
        cuda_rng = torch.cuda.get_rng_state_all()

    state = {
        "python_rng": python_rng,
        "numpy_rng": numpy_rng,
        "torch_rng": torch_rng,
    }
    if cuda_rng is not None:
        state["cuda_rng"] = cuda_rng

    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        safe_save_state(state, tmp_path)
        loaded_state = safe_load_state(tmp_path)
        
        assert_structures_equal(state, loaded_state)
        
        # Verify type identity explicitly
        assert type(loaded_state["python_rng"]) is tuple
        assert type(loaded_state["python_rng"][1]) is tuple  # nested MT state tuple
        assert type(loaded_state["numpy_rng"]) is tuple
        assert isinstance(loaded_state["numpy_rng"][1], np.ndarray)
        assert loaded_state["numpy_rng"][1].dtype == np.uint32
        assert type(loaded_state["torch_rng"]).__name__ == "Tensor"
        
        # Verify we can actually set the random states back
        random.setstate(loaded_state["python_rng"])
        np.random.set_state(loaded_state["numpy_rng"])
        torch.set_rng_state(loaded_state["torch_rng"])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(loaded_state["cuda_rng"])
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_composite_checkpoint_roundtrip():
    model = nn.Linear(5, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Step
    optimizer.step()
    scheduler.step()

    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "step": 42,
        "loss": 0.12345
    }

    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        safe_save_state(checkpoint, tmp_path)
        loaded = safe_load_state(tmp_path)
        
        assert_structures_equal(checkpoint, loaded)
        assert type(loaded["python_rng"]) is tuple
        assert type(loaded["numpy_rng"]) is tuple
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_general_validation_errors():
    class DummyObj:
        pass

    # Unsupported custom object
    with pytest.raises(TypeError):
        safe_save_state({"obj": DummyObj()}, "dummy.pt")

    # Unsupported function/lambda
    with pytest.raises(TypeError):
        safe_save_state({"fn": lambda x: x}, "dummy.pt")


def test_numpy_scalars_roundtrip():
    state = {
        "np_int": np.int64(42),
        "np_float": np.float32(3.14),
        "np_bool": np.bool_(True),
    }
    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        safe_save_state(state, tmp_path)
        loaded = safe_load_state(tmp_path)
        
        # Standalone scalars should be converted to native Python types on save
        assert type(loaded["np_int"]) is int
        assert loaded["np_int"] == 42
        
        assert type(loaded["np_float"]) is float
        assert abs(loaded["np_float"] - 3.14) < 1e-5
        
        assert type(loaded["np_bool"]) is bool
        assert loaded["np_bool"] is True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_numpy_dtype_validation():
    # complex64 is not in the closed supported list
    unsupported_arr = np.array([1+2j], dtype=np.complex64)
    with pytest.raises(TypeError) as excinfo:
        safe_save_state({"arr": unsupported_arr}, "dummy.pt")
    assert "Unsupported numpy dtype: complex64" in str(excinfo.value)
    
    # object array is not in the closed supported list
    obj_arr = np.array([{"a": 1}], dtype=object)
    with pytest.raises(TypeError):
        safe_save_state({"arr": obj_arr}, "dummy.pt")


def test_dict_key_validation():
    # float keys are not supported and raise TypeError
    with pytest.raises(TypeError):
        safe_save_state({3.14: "value"}, "dummy.pt")
        
    # tuple keys are not supported and raise TypeError
    with pytest.raises(TypeError):
        safe_save_state({("a", 1): "value"}, "dummy.pt")


def test_optimizer_wrapper_roundtrip():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    # Run steps to populate state
    x = torch.randn(5, 10)
    y = torch.randn(5, 2)
    loss = (model(x) - y).pow(2).sum()
    loss.backward()
    optimizer.step()

    orig_state = optimizer.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Save via wrapper
        safe_save_optimizer(orig_state, tmp_path)
        
        # Load via wrapper
        loaded_state = safe_load_optimizer(tmp_path)
        
        assert_structures_equal(orig_state, loaded_state)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_accelerate_shaped_state_roundtrip():
    model = nn.Linear(5, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    # Populate states
    x = torch.randn(5, 5)
    y = torch.randn(5, 2)
    loss = (model(x) - y).pow(2).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    accelerate_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }
    
    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        safe_save_state(accelerate_state, tmp_path)
        loaded = safe_load_state(tmp_path)
        
        assert_structures_equal(accelerate_state, loaded)
        
        # Verify exact type matching
        assert type(loaded["python_rng"]) is tuple
        assert type(loaded["numpy_rng"]) is tuple
        assert isinstance(loaded["numpy_rng"][1], np.ndarray)
        assert type(loaded["torch_rng"]).__name__ == "Tensor"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_fuzz_tag_system():
    # Tag says 'tuple' but type is not a list/value payload
    bad_tag_state = {
        "__metadata__": {"format_version": "1.1", "type": "general_state"},
        "__tensors__": {},
        "__scalars__": {},
        "__config__": {
            "bad_tuple": {
                "__pytype__": "tuple",
                "value": "not-a-list"  # malformed
            }
        }
    }
    
    header_bytes = json.dumps(bad_tag_state).encode('utf-8')
    padding_len = (8 - (8 + len(header_bytes)) % 8) % 8
    header_bytes += b' ' * padding_len
    header_len = len(header_bytes)
    
    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            
        with pytest.raises((TypeError, ValueError)):
            safe_load_state(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    # Unrecognized __pytype__ tag should be treated as a normal dict
    unrecognized_tag_state = {
        "__metadata__": {"format_version": "1.1", "type": "general_state"},
        "__tensors__": {},
        "__scalars__": {},
        "__config__": {
            "normal_dict": {
                "__pytype__": "unrecognized_custom_type",
                "some_data": 42
            }
        }
    }
    
    header_bytes = json.dumps(unrecognized_tag_state).encode('utf-8')
    padding_len = (8 - (8 + len(header_bytes)) % 8) % 8
    header_bytes += b' ' * padding_len
    header_len = len(header_bytes)
    
    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            
        loaded = safe_load_state(tmp_path)
        assert loaded["normal_dict"] == {
            "__pytype__": "unrecognized_custom_type",
            "some_data": 42
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_backward_compatibility_v1_0():
    # Construct a raw binary file representing a v1.0 optimizer checkpoint
    header = {
        "__metadata__": {
            "format_version": "1.0",
            "optimizer_type": "SGD"
        },
        "__tensors__": {
            "state.0.momentum_buffer": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8]
            }
        },
        "__scalars__": {
            "state.0.step": {
                "type": "int",
                "value": 42
            }
        },
        "__config__": {
            "param_groups": [{"lr": 0.01, "params": [0]}],
            "state_param_ids": [0]
        }
    }
    
    header_bytes = json.dumps(header).encode('utf-8')
    padding_len = (8 - (8 + len(header_bytes)) % 8) % 8
    header_bytes += b' ' * padding_len
    header_len = len(header_bytes)
    
    tensor_data = torch.tensor([1.5, -2.5], dtype=torch.float32)
    tensor_bytes = tensor_data.numpy().tobytes()
    
    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            f.write(tensor_bytes)
            
        # Load using safe_load_optimizer
        loaded_state = safe_load_optimizer(tmp_path)
        
        # Verify it loaded correctly
        assert loaded_state["param_groups"] == [{"lr": 0.01, "params": [0]}]
        assert loaded_state["state"][0]["step"] == 42
        assert torch.equal(loaded_state["state"][0]["momentum_buffer"], torch.tensor([1.5, -2.5], dtype=torch.float32))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_backward_compatibility_v1_0_fixture():
    # Load from the pre-generated static fixture file on disk
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "v1_optimizer.optimtensors")
    assert os.path.exists(fixture_path), f"Fixture not found at {fixture_path}"
    
    loaded_state = safe_load_optimizer(fixture_path)
    
    assert loaded_state["param_groups"] == [{"lr": 0.01, "params": [0]}]
    assert loaded_state["state"][0]["step"] == 42
    assert torch.equal(loaded_state["state"][0]["momentum_buffer"], torch.tensor([1.5, -2.5], dtype=torch.float32))


def test_atomic_write_consistency():
    import unittest.mock as mock
    
    state = {"a": 1}
    
    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Write initial file
        safe_save_state(state, tmp_path)
        
        # Mock os.replace to raise an exception, simulating a failure before rename
        with mock.patch("os.replace", side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                safe_save_state({"a": 2}, tmp_path)
                
        # Verify original file remains unmodified
        loaded = safe_load_state(tmp_path)
        assert loaded == {"a": 1}
        
        # Check atomic writes in safe_save_optimizer wrapper
        model = nn.Linear(5, 2)
        opt = torch.optim.Adam(model.parameters())
        opt_state = opt.state_dict()
        
        safe_save_optimizer(opt_state, tmp_path)
        
        with mock.patch("os.replace", side_effect=IOError("Disk full")):
            with pytest.raises(IOError):
                safe_save_optimizer(opt_state, tmp_path)
                
        # Verify optimizer file is still loadable
        loaded_opt = safe_load_optimizer(tmp_path)
        assert "param_groups" in loaded_opt
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
