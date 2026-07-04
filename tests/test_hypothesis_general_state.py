import os
import tempfile
import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st
from optimtensors import safe_save_state, safe_load_state
from optimtensors.serde import TORCH_TO_SAFETA, NUMPY_SUPPORTED_DTYPES

SUPPORTED_TORCH_DTYPES = list(TORCH_TO_SAFETA.keys())

# Strategy for shapes
shape_strategy = st.lists(st.integers(min_value=0, max_value=10), min_size=0, max_size=3).map(tuple)

def make_tensor(draw):
    shape = draw(shape_strategy)
    dtype = draw(st.sampled_from(SUPPORTED_TORCH_DTYPES))
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype)
    elif dtype == torch.bool:
        return torch.randint(0, 2, shape).to(dtype)
    else:
        return torch.randint(-100, 100, shape).to(dtype)

def make_ndarray(draw):
    shape = draw(shape_strategy)
    dtype_name = draw(st.sampled_from(list(NUMPY_SUPPORTED_DTYPES)))
    dtype = np.dtype(dtype_name)
    if dtype.kind in 'fc':  # float
        return np.random.normal(size=shape).astype(dtype)
    elif dtype.kind == 'b':  # bool
        return np.random.choice([True, False], size=shape).astype(dtype)
    else:  # int, uint
        return np.random.randint(0, 100, size=shape).astype(dtype)

# Strategy for standalone NumPy scalars
def make_numpy_scalar(draw):
    dt_name = draw(st.sampled_from(list(NUMPY_SUPPORTED_DTYPES)))
    dtype = np.dtype(dt_name)
    if dtype.kind in 'fc':  # float
        v = draw(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
    elif dtype.kind == 'b':  # bool
        v = draw(st.booleans())
    else:  # int, uint
        if dtype.name == 'int8':
            v = draw(st.integers(min_value=-128, max_value=127))
        elif dtype.name == 'uint8':
            v = draw(st.integers(min_value=0, max_value=255))
        elif dtype.name == 'int16':
            v = draw(st.integers(min_value=-32768, max_value=32767))
        elif dtype.name == 'uint16':
            v = draw(st.integers(min_value=0, max_value=65535))
        elif dtype.name == 'int32':
            v = draw(st.integers(min_value=-2147483648, max_value=2147483647))
        elif dtype.name == 'uint32':
            v = draw(st.integers(min_value=0, max_value=4294967295))
        elif dtype.name == 'int64':
            v = draw(st.integers(min_value=-9223372036854775808, max_value=9223372036854775807))
        elif dtype.name == 'uint64':
            v = draw(st.integers(min_value=0, max_value=18446744073709551615))
        else:
            v = draw(st.integers(min_value=0, max_value=100))
    return dtype.type(v)

numpy_scalar_strategy = st.composite(make_numpy_scalar)()

base_strategy = st.one_of(
    st.booleans(),
    st.integers(min_value=-10000, max_value=10000),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(min_size=0, max_size=20),
    st.none(),
    st.composite(make_tensor)(),
    st.composite(make_ndarray)(),
    numpy_scalar_strategy
)

def extend_strategy(children):
    return st.one_of(
        # Lists
        st.lists(children, min_size=0, max_size=5),
        # Tuples
        st.lists(children, min_size=0, max_size=5).map(tuple),
        # Dictionaries (only string/int keys)
        st.dictionaries(
            st.one_of(st.text(min_size=1, max_size=10), st.integers(min_value=0, max_value=10)),
            children,
            min_size=0,
            max_size=5
        )
    )

recursive_state_strategy = st.recursive(
    base_strategy,
    extend_strategy,
    max_leaves=15
)


def assert_structures_equal_with_types(val1, val2):
    """Recursively checks that two nested structures are equal, asserting exact type matches."""
    # Handle custom NumPy scalar conversion during save (numpy scalar -> python scalar)
    if getattr(type(val1), "__module__", "") == "numpy" and type(val1).__name__ != "ndarray":
        # val1 is a NumPy scalar, which was converted to a native Python type on save
        val1 = val1.item()

    assert type(val1) is type(val2), f"Type mismatch: {type(val1)} != {type(val2)} for values {val1!r} and {val2!r}"

    if isinstance(val1, dict):
        assert set(val1.keys()) == set(val2.keys())
        for k in val1:
            assert_structures_equal_with_types(val1[k], val2[k])
    elif isinstance(val1, list):
        assert len(val1) == len(val2)
        for x, y in zip(val1, val2):
            assert_structures_equal_with_types(x, y)
    elif isinstance(val1, tuple):
        assert len(val1) == len(val2)
        for x, y in zip(val1, val2):
            assert_structures_equal_with_types(x, y)
    elif type(val1).__name__ == "Tensor":
        assert val1.shape == val2.shape
        assert val1.dtype == val2.dtype
        if val1.numel() > 0:
            nan_mask1 = val1.isnan()
            nan_mask2 = val2.isnan()
            assert torch.equal(nan_mask1, nan_mask2)
            assert torch.equal(val1.nan_to_num(nan=0.0), val2.nan_to_num(nan=0.0))
    elif isinstance(val1, np.ndarray):
        assert val1.shape == val2.shape
        assert val1.dtype == val2.dtype
        assert np.array_equal(val1, val2)
    else:
        assert val1 == val2


@settings(max_examples=100, deadline=None)
@given(state=recursive_state_strategy)
def test_hypothesis_general_state_roundtrip(state):
    # Ensure the root is always a dictionary, as required by the API
    root_state = {"root": state}

    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        safe_save_state(root_state, tmp_path)
        loaded = safe_load_state(tmp_path)
        
        assert_structures_equal_with_types(root_state, loaded)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
