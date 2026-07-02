import os
import tempfile
import pytest
import torch
from hypothesis import given, settings, strategies as st
from optimtensors.serde import safe_save_optimizer, safe_load_optimizer, safe_load_into_optimizer, TORCH_TO_SAFETA

# Supported dtypes
SUPPORTED_DTYPES = list(TORCH_TO_SAFETA.keys())

# Strategy for shapes
shape_strategy = st.lists(st.integers(min_value=0, max_value=20), min_size=0, max_size=4).map(tuple)
large_shape_strategy = st.sampled_from([
    (),
    (0,),
    (100000,),
    (2, 0, 5),
    (10, 5, 2, 2)
])
final_shape_strategy = st.one_of(shape_strategy, large_shape_strategy)

# Strategy for scalars
scalar_strategy = st.one_of(
    st.integers(min_value=-1000000, max_value=1000000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(min_size=0, max_size=50),
    st.none(),
    st.lists(st.integers(min_value=-100, max_value=100), min_size=0, max_size=10),
    st.lists(st.lists(st.integers(min_value=-10, max_value=10), min_size=0, max_size=5), min_size=0, max_size=5)
)

def tensor_equal(t1, t2):
    if t1.shape != t2.shape or t1.dtype != t2.dtype:
        return False
    if t1.numel() == 0:
        return True
    nan_mask1 = t1.isnan()
    nan_mask2 = t2.isnan()
    if not torch.equal(nan_mask1, nan_mask2):
        return False
    return torch.equal(t1.nan_to_num(nan=0.0), t2.nan_to_num(nan=0.0))

@settings(max_examples=200, deadline=None)
@given(
    shapes=st.lists(final_shape_strategy, min_size=1, max_size=5),
    dtypes=st.lists(st.sampled_from(SUPPORTED_DTYPES), min_size=1, max_size=5),
    scalars=st.dictionaries(st.text(min_size=1, max_size=10), scalar_strategy, min_size=0, max_size=5)
)
def test_hypothesis_serde_roundtrip(shapes, dtypes, scalars):
    num_params = len(shapes)
    dtypes_matched = [dtypes[i % len(dtypes)] for i in range(num_params)]
    
    state = {}
    for i, (shape, dtype) in enumerate(zip(shapes, dtypes_matched)):
        p_state = {}
        if dtype.is_floating_point:
            t = torch.randn(shape, dtype=dtype)
        elif dtype == torch.bool:
            t = torch.randint(0, 2, shape).to(dtype)
        else:
            t = torch.randint(-100, 100, shape).to(dtype)
            
        p_state["exp_avg"] = t
        
        if i % 3 == 0:
            p_state["history"] = [t.clone(), None, t.clone()]
            
        for k, v in scalars.items():
            p_state[k] = v
            
        state[i] = p_state
        
    state_dict = {
        "state": state,
        "param_groups": [{"params": list(range(num_params)), "lr": 0.01}]
    }
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(state_dict, tmp_path)
        loaded = safe_load_optimizer(tmp_path)
        
        assert len(loaded["state"]) == len(state_dict["state"])
        for p_id in state_dict["state"]:
            orig_p = state_dict["state"][p_id]
            loaded_p = loaded["state"][p_id]
            assert set(orig_p.keys()) == set(loaded_p.keys())
            
            for k in orig_p:
                v_orig = orig_p[k]
                v_load = loaded_p[k]
                
                if isinstance(v_orig, torch.Tensor):
                    assert isinstance(v_load, torch.Tensor)
                    assert tensor_equal(v_orig, v_load)
                elif isinstance(v_orig, (list, tuple)) and len(v_orig) > 0 and any(isinstance(x, torch.Tensor) for x in v_orig):
                    assert isinstance(v_load, (list, tuple))
                    assert len(v_orig) == len(v_load)
                    for x_orig, x_load in zip(v_orig, v_load):
                        if x_orig is None:
                            assert x_load is None
                        else:
                            assert isinstance(x_load, torch.Tensor)
                            assert tensor_equal(x_orig, x_load)
                else:
                    assert v_orig == v_load
                    
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@settings(max_examples=200, deadline=None)
@given(
    num_params=st.sampled_from([1, 2, 50, 500]),
    scalar_val=scalar_strategy,
    dtype=st.sampled_from(SUPPORTED_DTYPES)
)
def test_hypothesis_num_params(num_params, scalar_val, dtype):
    state = {}
    for i in range(num_params):
        shape = (1,) if num_params > 2 else (2, 3)
        if dtype.is_floating_point:
            t = torch.randn(shape, dtype=dtype)
        elif dtype == torch.bool:
            t = torch.randint(0, 2, shape).to(dtype)
        else:
            t = torch.randint(-10, 10, shape).to(dtype)
            
        state[i] = {
            "exp_avg": t,
            "scalar_key": scalar_val
        }
        
    state_dict = {
        "state": state,
        "param_groups": [{"params": list(range(num_params)), "lr": 0.01}]
    }
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        safe_save_optimizer(state_dict, tmp_path)
        loaded = safe_load_optimizer(tmp_path)
        assert len(loaded["state"]) == num_params
        for i in range(num_params):
            assert tensor_equal(state_dict["state"][i]["exp_avg"], loaded["state"][i]["exp_avg"])
            assert state_dict["state"][i]["scalar_key"] == loaded["state"][i]["scalar_key"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
