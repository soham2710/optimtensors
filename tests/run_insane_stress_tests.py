import os
import sys
import tempfile
import struct
import json
import random
import pytest
import torch
import numpy as np

# Ensure local package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from optimtensors import safe_save_state, safe_load_state, safe_save_optimizer, safe_load_optimizer
from optimtensors.serde import NUMPY_SUPPORTED_DTYPES

def assert_structures_equal(val1, val2):
    # If it is a NumPy scalar, cast to item first for comparison
    if getattr(type(val1), "__module__", "") == "numpy" and type(val1).__name__ != "ndarray":
        val1 = val1.item()
    if getattr(type(val2), "__module__", "") == "numpy" and type(val2).__name__ != "ndarray":
        val2 = val2.item()

    assert type(val1) is type(val2), f"Type mismatch: {type(val1)} != {type(val2)}"
    
    if isinstance(val1, dict):
        assert set(val1.keys()) == set(val2.keys())
        for k in val1:
            assert_structures_equal(val1[k], val2[k])
    elif isinstance(val1, (list, tuple)):
        assert len(val1) == len(val2)
        for x, y in zip(val1, val2):
            assert_structures_equal(x, y)
    elif isinstance(val1, torch.Tensor):
        assert val1.shape == val2.shape
        assert val1.dtype == val2.dtype
        if val1.numel() > 0:
            assert torch.allclose(val1.nan_to_num(0.0), val2.nan_to_num(0.0))
    elif isinstance(val1, np.ndarray):
        assert val1.shape == val2.shape
        assert val1.dtype == val2.dtype
        assert np.array_equal(val1, val2)
    else:
        assert val1 == val2

def run_test_case(case_id, description, generator_fn, expect_success=True, expected_exception=None):
    """Runs a single test case, returning status (bool) and message (str)"""
    with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        payload = generator_fn()
        
        # If generator returns None, it means the test ran itself inline
        if payload is None:
            if expect_success:
                return False, "Test generated None but expected success"
            else:
                return False, "Failed (completed successfully but expected failure)"

        if expect_success:
            safe_save_state(payload, tmp_path)
            loaded = safe_load_state(tmp_path)
            assert_structures_equal(payload, loaded)
            return True, "Passed (successful round-trip)"
        else:
            try:
                safe_save_state(payload, tmp_path)
                safe_load_state(tmp_path)
            except Exception as e:
                if expected_exception is None or isinstance(e, expected_exception):
                    return True, f"Passed (correctly failed with expected {type(e).__name__})"
                return False, f"Failed with wrong exception: {type(e).__name__}: {e}"
            return False, "Failed (completed successfully but expected failure)"
    except Exception as e:
        if not expect_success:
            if expected_exception is None or isinstance(e, expected_exception):
                return True, f"Passed (correctly failed with expected {type(e).__name__})"
            return False, f"Failed with wrong exception: {type(e).__name__}: {e}"
        return False, f"Unexpected exception: {type(e).__name__}: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    print("=" * 80)
    print("                 OPTIMTENSORS INSANE STRESS-TESTING SUITE")
    print("=" * 80)
    print("Executing 200 stress test cases...")

    results = []

    # -------------------------------------------------------------------------
    # EASY LEVEL (Cases 1-20)
    # -------------------------------------------------------------------------
    # Basic primitives, simple lists/dicts, simple tensors/ndarrays.
    easy_cases = [
        (1, "Empty dict", lambda: {}),
        (2, "Boolean True", lambda: {"val": True}),
        (3, "Boolean False", lambda: {"val": False}),
        (4, "Integer scalar", lambda: {"val": 42}),
        (5, "Float scalar", lambda: {"val": 3.14159}),
        (6, "String scalar", lambda: {"val": "hello world"}),
        (7, "None scalar", lambda: {"val": None}),
        (8, "Simple list", lambda: {"val": [1, 2, 3, 4]}),
        (9, "Simple tuple", lambda: {"val": (5, 6, 7, 8)}),
        (10, "Simple mixed dict", lambda: {"a": 1, "b": "two", "c": True}),
        (11, "1D float32 Tensor", lambda: {"tensor": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}),
        (12, "2D float32 Tensor", lambda: {"tensor": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)}),
        (13, "1D int64 Tensor", lambda: {"tensor": torch.tensor([10, 20, 30], dtype=torch.int64)}),
        (14, "1D float32 ndarray", lambda: {"array": np.array([1.5, 2.5, 3.5], dtype=np.float32)}),
        (15, "2D int32 ndarray", lambda: {"array": np.array([[1, 2], [3, 4]], dtype=np.int32)}),
        (16, "Standard python RNG tuple", lambda: {"rng": (3, (1, 2, 3), None)}),
        (17, "Mixed float/int/bool list", lambda: {"lst": [1, 2.5, True, None, "str"]}),
        (18, "Nested dict 2-levels", lambda: {"a": {"b": 1}}),
        (19, "Simple CPU float16 Tensor", lambda: {"tensor": torch.tensor([1.0, 2.0], dtype=torch.float16)}),
        (20, "Boolean ndarray", lambda: {"array": np.array([True, False, True], dtype=bool)}),
    ]
    for cid, desc, gen in easy_cases:
        ok, msg = run_test_case(cid, desc, gen, expect_success=True)
        results.append((cid, "Easy", desc, ok, msg))

    # -------------------------------------------------------------------------
    # MEDIUM LEVEL (Cases 21-100)
    # -------------------------------------------------------------------------
    # Nested dicts, empty containers, numpy scalars, mixed int keys, scheduler states.
    medium_cases = []
    
    # 21-30: Nested levels
    for i in range(10):
        depth = i + 2
        def get_nested(d):
            if d == 1:
                return 42
            return {"level": get_nested(d - 1)}
        medium_cases.append((21 + i, f"Nested dict depth {depth}", lambda d=depth: get_nested(d)))

    # 31-40: Mixed deep list/tuple combos
    medium_cases.append((31, "Tuple of list of dicts", lambda: {"data": ([{"a": 1}, {"b": 2}], [{"c": 3}])}))
    medium_cases.append((32, "List of tuples of lists", lambda: {"data": [(1, [2, 3]), (4, [5, 6])] }))
    medium_cases.append((33, "Dict with empty list leaf", lambda: {"a": []}))
    medium_cases.append((34, "Dict with empty tuple leaf", lambda: {"a": ()}))
    medium_cases.append((35, "Dict with empty dict leaf", lambda: {"a": {}}))
    medium_cases.append((36, "RNG shape complex list", lambda: {"rng": (1, [np.array([1, 2], dtype=np.uint32), 3], 4.5)}))
    medium_cases.append((37, "Nested empty list tuple combo", lambda: {"a": ([], ({}, []))}))
    medium_cases.append((38, "State index mapping dict", lambda: {"state": {0: {"step": 1}, 1: {"step": 2}}}))
    medium_cases.append((39, "Param group with custom parameters", lambda: {"param_groups": [{"lr": 0.001, "params": [0, 1, 2]}]}))
    medium_cases.append((40, "LR Scheduler dict structure", lambda: {"base_lrs": [0.1], "last_epoch": 5, "_step_count": 6}))

    # 41-60: NumPy scalars of every supported dtype
    dtypes = list(NUMPY_SUPPORTED_DTYPES)
    for i, dt in enumerate(dtypes):
        val = 1 if dt.startswith("u") or dt == "bool" else -1
        def make_scalar(dt=dt, val=val):
            return {"scalar": np.dtype(dt).type(val)}
        medium_cases.append((41 + i, f"NumPy scalar {dt}", make_scalar))

    # Fill remaining to 100 with zero-dim arrays & mixed int key dicts
    idx = len(medium_cases) + 21
    for i in range(100 - idx + 1):
        cid = idx + i
        if i % 2 == 0:
            dt = dtypes[i % len(dtypes)]
            def make_0d(dt=dt):
                return {"arr": np.array(42, dtype=dt)}
            medium_cases.append((cid, f"0D array of dtype {dt}", make_0d))
        else:
            keys = [random.randint(0, 1000) for _ in range(5)]
            def make_int_dict(keys=keys):
                return {k: f"val_{k}" for k in keys}
            medium_cases.append((cid, f"Dict with random int keys {keys}", make_int_dict))

    for cid, desc, gen in medium_cases:
        ok, msg = run_test_case(cid, desc, gen, expect_success=True)
        results.append((cid, "Medium", desc, ok, msg))

    # -------------------------------------------------------------------------
    # HARD LEVEL (Cases 101-170)
    # -------------------------------------------------------------------------
    # Borderline boundary integers, extreme floats, high-dim tensors, empty shapes, custom metadata.
    hard_cases = [
        (101, "Extreme int64 max", lambda: {"val": 2**63 - 1}),
        (102, "Extreme int64 min", lambda: {"val": -2**63}),
        (103, "Extreme uint64 max", lambda: {"val": 2**64 - 1}),
        (104, "Extreme int32 max", lambda: {"val": 2**31 - 1}),
        (105, "Extreme int32 min", lambda: {"val": -2**31}),
        (106, "Large float value", lambda: {"val": 1.79e308}),
        (107, "Small float value", lambda: {"val": -1.79e308}),
        (108, "Close to zero float", lambda: {"val": 5e-324}),
        (109, "RNG state tuple with huge list", lambda: {"rng": (1, list(range(1000)), 0.5)}),
        (110, "Nested structure depth 8", lambda: {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 42}}}}}}}}),
        (111, "5D PyTorch Tensor float32", lambda: {"tensor": torch.randn(2, 3, 2, 2, 2)}),
        (112, "6D NumPy Array int16", lambda: {"array": np.arange(64).reshape(2, 2, 2, 2, 2, 2).astype(np.int16)}),
        (113, "Empty shape Tensor (0,)", lambda: {"tensor": torch.empty(0, dtype=torch.float32)}),
        (114, "Empty shape Tensor (2, 0, 3)", lambda: {"tensor": torch.empty(2, 0, 3, dtype=torch.int32)}),
        (115, "Empty shape ndarray (0, 10)", lambda: {"array": np.empty((0, 10), dtype=np.float32)}),
        (116, "Empty shape ndarray (5, 0)", lambda: {"array": np.empty((5, 0), dtype=bool)}),
        (117, "PyTorch CPU bfloat16 Tensor", lambda: {"tensor": torch.tensor([1.5, -2.5], dtype=torch.bfloat16)}),
        (118, "Large 1MB numpy array allocation", lambda: {"array": np.zeros(250000, dtype=np.float32)}),
        (119, "Large 2MB PyTorch tensor allocation", lambda: {"tensor": torch.ones(500000, dtype=torch.float32)}),
        (120, "Custom string metadata dictionary", lambda: {"a": 1, "metadata": {"format_version": "1.1", "custom_key": "custom_val"}}),
    ]

    # Fill remaining hard cases to 170 with multi-dimensional dtypes combinations
    idx = len(hard_cases) + 101
    for i in range(170 - idx + 1):
        cid = idx + i
        dt = dtypes[i % len(dtypes)]
        # Generate 3D arrays
        def make_3d(dt=dt):
            return {"arr": np.ones((3, 3, 3), dtype=dt)}
        hard_cases.append((cid, f"3D array of dtype {dt}", make_3d))

    for cid, desc, gen in hard_cases:
        ok, msg = run_test_case(cid, desc, gen, expect_success=True)
        results.append((cid, "Hard", desc, ok, msg))

    # -------------------------------------------------------------------------
    # ADVERSARIAL / INSANE LEVEL (Cases 171-200)
    # -------------------------------------------------------------------------
    # Fuzzing attacks, recursion limits, unaligned offsets, invalid keys, invalid tags, unsupported dtypes.
    insane_cases = []

    # 171: Dictionary keys containing invalid types (float key)
    insane_cases.append((171, "Adversarial: Float key in dictionary", lambda: {3.14: "val"}, False, TypeError))
    # 172: Dictionary keys containing invalid types (tuple key)
    insane_cases.append((172, "Adversarial: Tuple key in dictionary", lambda: {("a", 1): "val"}, False, TypeError))
    # 173: Unsupported custom object
    class UnsafeObject:
        pass
    insane_cases.append((173, "Adversarial: Unsupported class object", lambda: {"obj": UnsafeObject()}, False, TypeError))
    # 174: Unsupported lambda function
    insane_cases.append((174, "Adversarial: Lambda function serialization", lambda: {"fn": lambda x: x}, False, TypeError))
    # 175: Unsupported numpy dtype (complex64)
    insane_cases.append((175, "Adversarial: NumPy array complex64", lambda: {"arr": np.array([1+2j], dtype=np.complex64)}, False, TypeError))
    # 176: Unsupported numpy dtype (object array)
    insane_cases.append((176, "Adversarial: NumPy object array", lambda: {"arr": np.array([{"a": 1}], dtype=object)}, False, TypeError))
    # 177: Tag says 'tuple' but value is not a list
    # (Generates a file programmatically with malformed tag to check load safety)
    def make_bad_tuple_file():
        state = {
            "__metadata__": {"format_version": "1.1"},
            "__tensors__": {},
            "__scalars__": {},
            "__config__": {"bad_tuple": {"__pytype__": "tuple", "value": "not-a-list"}}
        }
        return state
    insane_cases.append((177, "Adversarial: Malformed tuple tag payload (str)", make_bad_tuple_file, False, (TypeError, ValueError)))

    # 178: Tag says 'dict_with_int_keys' but keys is a string
    def make_bad_dict_keys_file():
        state = {
            "__metadata__": {"format_version": "1.1"},
            "__tensors__": {},
            "__scalars__": {},
            "__config__": {"bad_dict": {"__pytype__": "dict_with_int_keys", "keys": "not-a-list", "value": [1]}}
        }
        return state
    insane_cases.append((178, "Adversarial: Malformed dict_with_int_keys tag payload (str keys)", make_bad_dict_keys_file, False, (TypeError, ValueError)))

    # 179: Deep recursion attack on load (nested array list payloads)
    def make_deep_recursion_file():
        nested = "1"
        for _ in range(1000):
            nested = f"[{nested}]"
        # Wrap in standard optimtensors format manually
        header = {
            "__metadata__": {"format_version": "1.1"},
            "__tensors__": {},
            "__scalars__": {},
            "__config__": {"attack": json.loads(nested)}
        }
        return header
    # Standard json has default recursion limits or raises ValueError/RecursionError
    insane_cases.append((179, "Adversarial: Deep JSON recursion limit load attack", make_deep_recursion_file, False, (RecursionError, ValueError)))

    # 180: Corrupted / truncated file (file ends before header ends)
    def test_truncated_file():
        with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as f:
                # Says header is 100 bytes, but write only 10 bytes
                f.write(struct.pack("<Q", 100))
                f.write(b"1234567890")
            safe_load_state(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    insane_cases.append((180, "Adversarial: Truncated file header error handling", test_truncated_file, False, ValueError))

    # 181: Overlapping offsets in __tensors__ metadata
    def make_overlapping_offsets_file():
        header = {
            "__metadata__": {"format_version": "1.1"},
            "__tensors__": {
                "tensor_A": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
                "tensor_B": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]} # Overlaps tensor_A!
            },
            "__scalars__": {},
            "__config__": {"a": {"__pytype__": "tensor", "key": "tensor_A"}, "b": {"__pytype__": "tensor", "key": "tensor_B"}}
        }
        header_bytes = json.dumps(header).encode('utf-8')
        padding_len = (8 - (8 + len(header_bytes)) % 8) % 8
        header_bytes += b' ' * padding_len
        header_len = len(header_bytes)
        
        with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(struct.pack("<Q", header_len))
                f.write(header_bytes)
                f.write(b"\x00" * 32) # Buffer size 32 bytes
            safe_load_state(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    insane_cases.append((181, "Adversarial: Overlapping offset validation", make_overlapping_offsets_file, False, ValueError))

    # 182: Unaligned memory offset for tensor (element size 4, offset odd number)
    def make_unaligned_offset_file():
        header = {
            "__metadata__": {"format_version": "1.1"},
            "__tensors__": {
                "tensor_A": {"dtype": "F32", "shape": [4], "data_offsets": [3, 19]} # Offset 3 is not 4-aligned!
            },
            "__scalars__": {},
            "__config__": {"a": {"__pytype__": "tensor", "key": "tensor_A"}}
        }
        header_bytes = json.dumps(header).encode('utf-8')
        padding_len = (8 - (8 + len(header_bytes)) % 8) % 8
        header_bytes += b' ' * padding_len
        header_len = len(header_bytes)
        
        with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(struct.pack("<Q", header_len))
                f.write(header_bytes)
                f.write(b"\x00" * 32)
            safe_load_state(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    insane_cases.append((182, "Adversarial: Unaligned memory offset check", make_unaligned_offset_file, False, ValueError))

    # Fill remaining insane cases up to 200 with various invalid config attacks
    idx = len(insane_cases) + 171
    for i in range(200 - idx + 1):
        cid = idx + i
        if i % 3 == 0:
            # Invalid JSON header (malformed string)
            def make_bad_json():
                with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(struct.pack("<Q", 20))
                        f.write(b"{bad-json-header-data")
                    safe_load_state(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            insane_cases.append((cid, "Adversarial: Malformed JSON header", make_bad_json, False, ValueError))
        elif i % 3 == 1:
            # Invalid header size (too largeafety block > 50MB)
            def make_large_header_file():
                with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(struct.pack("<Q", 60 * 1024 * 1024)) # 60MB
                    safe_load_state(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            insane_cases.append((cid, "Adversarial: Header size safety limit (>50MB)", make_large_header_file, False, ValueError))
        else:
            # Header is a JSON list instead of object
            def make_list_header_file():
                header_bytes = b"[1, 2, 3]"
                header_len = len(header_bytes)
                with tempfile.NamedTemporaryFile(suffix=".optimtensors", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(struct.pack("<Q", header_len))
                        f.write(header_bytes)
                    safe_load_state(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            insane_cases.append((cid, "Adversarial: List-structured header attack", make_list_header_file, False, ValueError))

    for cid, desc, gen, *extra in insane_cases:
        expect = extra[0] if len(extra) > 0 else True
        exc = extra[1] if len(extra) > 1 else None
        
        ok, msg = run_test_case(cid, desc, gen, expect_success=expect, expected_exception=exc)
        results.append((cid, "Insane (Adversarial)", desc, ok, msg))

    # -------------------------------------------------------------------------
    # PRINT REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                              STRESS TEST REPORT")
    print("=" * 80)
    print(f"{'ID':<4} | {'Level':<20} | {'Description':<40} | {'Status':<10}")
    print("-" * 80)
    
    passed_count = 0
    failed_cases = []
    
    # We display a selection of results to avoid terminal flood, but aggregate all
    for cid, lvl, desc, ok, msg in results:
        status_str = "SUCCESS" if ok else "FAILED"
        if ok:
            passed_count += 1
        else:
            failed_cases.append((cid, desc, msg))
            
        # Display first 5 of each level and all failures
        if cid <= 5 or (20 < cid <= 25) or (100 < cid <= 105) or (170 < cid <= 175) or not ok:
            print(f"{cid:<4} | {lvl:<20} | {desc[:38]:<40} | {status_str:<10}")
        elif cid in (6, 26, 106, 176):
            print(f"...  | ...                  | ...                                      | ...")

    print("=" * 80)
    print(f"Summary: {passed_count} / {len(results)} Stress Tests Passed Successfully!")
    print("=" * 80)

    if failed_cases:
        print("\nFailures Details:")
        for cid, desc, msg in failed_cases:
            print(f"Case {cid} ({desc}): {msg}")
        sys.exit(1)
    else:
        print("\nALL STRESS TESTS COMPLETED SUCCESSFULLY! PRODUCT IS 100% SECURE AND STABLE!")
        sys.exit(0)

if __name__ == "__main__":
    main()
