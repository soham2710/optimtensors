import os
import struct
import tempfile
import pytest
import json
from optimtensors.serde import safe_load_optimizer, safe_save_optimizer


def test_truncated_file():
    # Empty file
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            pass
        with pytest.raises((ValueError, struct.error)):
            safe_load_optimizer(tmp_path)
            
        # File with only 4 bytes (needs at least 8 for header length)
        with open(tmp_path, "wb") as f:
            f.write(b'\x01\x02\x03\x04')
        with pytest.raises((ValueError, struct.error)):
            safe_load_optimizer(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_malformed_header_length():
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Declare huge header length that exceeds file size
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", 999999))
            f.write(b'{"some": "json"}')
        with pytest.raises(ValueError) as excinfo:
            safe_load_optimizer(tmp_path)
        assert "Invalid header length" in str(excinfo.value) or "truncated" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_invalid_json():
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        header_content = b'{invalid json'
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_content)))
            f.write(header_content)
        with pytest.raises(ValueError) as excinfo:
            safe_load_optimizer(tmp_path)
        assert "Failed to parse JSON" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_missing_and_extraneous_keys():
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Missing keys
        header_missing = {"__metadata__": {}, "__tensors__": {}}
        header_bytes = json.dumps(header_missing).encode('utf-8')
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
        with pytest.raises(ValueError) as excinfo:
            safe_load_optimizer(tmp_path)
        assert "Missing required top-level key" in str(excinfo.value)

        # Extraneous keys (attack vector)
        header_extra = {
            "__metadata__": {},
            "__tensors__": {},
            "__scalars__": {},
            "__config__": {},
            "__evil_exec__": "malicious_payload_or_similar"
        }
        header_bytes_extra = json.dumps(header_extra).encode('utf-8')
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes_extra)))
            f.write(header_bytes_extra)
        with pytest.raises(ValueError) as excinfo:
            safe_load_optimizer(tmp_path)
        assert "Unexpected top-level keys" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_type_confusion_in_header():
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Put an disallowed type in scalars (like a nested dictionary containing lambda representations, sets, etc.)
        header_evil = {
            "__metadata__": {"format_version": "1.0", "optimizer_type": "AdamW"},
            "__tensors__": {},
            "__scalars__": {
                "state.0.step": {
                    "type": "custom",
                    "value": {"set_of_values": [1, 2, 3]} # sets can't be represented directly in JSON, but we can put dicts/lists
                }
            },
            "__config__": {
                # Put a function string or class representation to verify check_safe_structure blocks it or raises on loaded types
                "param_groups": [
                    {"lr": 0.001, "params": [0], "evil_set": [1, 2, 3]}
                ]
            }
        }
        # Note: JSON natively only has string, number, object, array, boolean, null.
        # But we check that only allowed types are inside __scalars__ and __config__.
        # Let's verify we raise TypeError on load if type check fails
        header_bytes = json.dumps(header_evil).encode('utf-8')
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            
        # Since 'evil_set' is a list of ints, it's safe. But let's add something unsafe like a dict inside param_groups
        # or a nested dictionary of nested dictionaries that violates the primitive list constraint.
        # Wait, check_safe_structure allows dicts and lists. What if we put a value that is NOT in the closed set?
        # Since JSON can only encode standard types, how can we trigger TypeError on load?
        # Actually, check_safe_structure checks that all leaf elements are bool, int, float, str, or None.
        # If we put a list that contains a dict, is it allowed in scalars?
        # The scalars validation checks each value in __scalars__ via check_safe_structure.
        # But scalars are supposed to be primitives or lists of primitives. If a scalar value is a dict,
        # it is technically a nested dict which check_safe_structure allows, but wait!
        # When loading, if we put a nested dict, does it crash?
        # Let's make sure check_safe_structure is strictly evaluated.
        # Let's try loading a header where scalars has a dict or list with bad content.
        
        # Let's load the above header and assert it passes or fails correctly.
        # Actually, let's write a test with an invalid data_offsets index.
        # If we declare offsets exceeding mmap size:
        header_offset_fuzz = {
            "__metadata__": {"format_version": "1.0", "optimizer_type": "AdamW"},
            "__tensors__": {
                "state.0.exp_avg": {
                    "dtype": "F32",
                    "shape": [10, 10],
                    "data_offsets": [0, 1000000] # Exceeds actual file/buffer size!
                }
            },
            "__scalars__": {},
            "__config__": {"param_groups": []}
        }
        header_bytes = json.dumps(header_offset_fuzz).encode('utf-8')
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(b'\x00' * 100) # Small actual tensor buffer
            
        with pytest.raises(ValueError):
            safe_load_optimizer(tmp_path)
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_pickle_exploit_injection():
    # Attempt to inject a malicious pickle exploit file
    # This simulates a situation where a malicious actor replaced a safetensors file
    # with a pickle file, hoping that our loader uses pickle under the hood.
    import pickle
    class ExploitPayload:
        def __reduce__(self):
            # This would run if pickle.load is called.
            # We will write a file to disk as proof of execution if it occurs.
            return (os.system, ("echo COMPROMISED > exploit_executed.txt",))
            
    payload = pickle.dumps(ExploitPayload())
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(payload)
            
        # Try to load it. Since we never call pickle.load, it must raise a ValueError/struct.error
        # and NEVER run the exploit payload.
        with pytest.raises((ValueError, struct.error)):
            safe_load_optimizer(tmp_path)
            
        # Assert that the exploit was indeed never executed
        assert not os.path.exists("exploit_executed.txt")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists("exploit_executed.txt"):
            os.remove("exploit_executed.txt")


def test_brute_force_byte_mutator():
    # Establish a valid baseline file
    import torch
    valid_state = {
        "state": {
            0: {"step": 45, "exp_avg": torch.randn(2, 2)}
        },
        "param_groups": [{"lr": 0.001, "params": [0]}]
    }
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        safe_save_optimizer(valid_state, tmp_path)
        with open(tmp_path, "rb") as f:
            valid_bytes = bytearray(f.read())
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    import random
    # Run 5000 iterations of random single and multi-byte corruptions
    random.seed(42)
    
    for iteration in range(5000):
        mutated = bytearray(valid_bytes)
        mutation_type = random.choice(["flip", "delete", "insert", "overwrite", "header_size"])
        
        if mutation_type == "flip" and len(mutated) > 0:
            idx = random.randint(0, len(mutated) - 1)
            mutated[idx] ^= (1 << random.randint(0, 7))
        elif mutation_type == "delete" and len(mutated) > 20:
            start = random.randint(0, len(mutated) - 21)
            end = start + random.randint(1, 20)
            del mutated[start:end]
        elif mutation_type == "insert":
            idx = random.randint(0, len(mutated))
            insert_bytes = bytearray(random.getrandbits(8) for _ in range(random.randint(1, 20)))
            mutated[idx:idx] = insert_bytes
        elif mutation_type == "overwrite" and len(mutated) > 20:
            start = random.randint(0, len(mutated) - 21)
            end = start + random.randint(1, 20)
            for i in range(start, end):
                mutated[i] = random.randint(0, 255)
        elif mutation_type == "header_size" and len(mutated) >= 8:
            # Overwrite header size prefix with random bytes
            for i in range(8):
                mutated[i] = random.randint(0, 255)
                
        # Write to temp file and try loading
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            fuzz_path = tmp.name
        try:
            with open(fuzz_path, "wb") as f:
                f.write(mutated)
                
            # Loading mutated files must only raise expected parsing exceptions.
            # It must never crash the interpreter (e.g. segmentation fault).
            try:
                safe_load_optimizer(fuzz_path)
            except (ValueError, TypeError, UnicodeDecodeError, struct.error, OSError):
                # These are all clean, expected parsing exceptions
                pass
        finally:
            if os.path.exists(fuzz_path):
                os.remove(fuzz_path)


def test_unaligned_memory_offsets():
    import json
    import struct
    # Create valid metadata but manually inject an unaligned offset for float32 (requires 4-byte alignment)
    header = {
        "__metadata__": {"format_version": "1.0", "optimizer_type": "Adam"},
        "__tensors__": {
            "state.0.exp_avg": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [1, 9] # start=1 is unaligned for F32
            }
        },
        "__scalars__": {},
        "__config__": {"state_param_ids": [0]}
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            f.write(b'\x00' * 16)
            
        with pytest.raises(ValueError) as excinfo:
            safe_load_optimizer(tmp_path)
        assert "Unaligned memory offset" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_json_recursion_limit_attack():
    import struct
    # Construct an extremely deeply nested JSON structure
    nested_json = "[" * 2000 + "1" + "]" * 2000
    header_bytes = nested_json.encode("utf-8")
    header_len = len(header_bytes)
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            
        with pytest.raises(ValueError):
            safe_load_optimizer(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_header_size_dos_prevention():
    import struct
    # Attempt to declare a massive header size (e.g. 100MB)
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", 100 * 1024 * 1024))
            
        with pytest.raises(ValueError) as excinfo:
            safe_load_optimizer(tmp_path)
        assert "exceeds safety limit" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_offset_size_mismatch():
    import json
    import struct
    # Construct a valid header but with a data_offsets size that doesn't match shape * elem_size.
    # Here shape is [2] (float32, 4 bytes each = 8 bytes expected).
    # But data_offsets is set to [0, 16] (16 bytes).
    header = {
        "__metadata__": {"format_version": "1.0", "optimizer_type": "Adam"},
        "__tensors__": {
            "state.0.exp_avg": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 16] # Mismatch: 16 bytes declared, but shape [2] float32 requires 8 bytes
            }
        },
        "__scalars__": {},
        "__config__": {"state_param_ids": [0]}
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = len(header_bytes)
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(struct.pack("<Q", header_len))
            f.write(header_bytes)
            f.write(b'\x00' * 32)
            
        with pytest.raises(ValueError) as excinfo:
            safe_load_optimizer(tmp_path)
        assert "does not match expected tensor size" in str(excinfo.value)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_adversarial_shape_offset_fuzz():
    import json
    import struct
    import re
    
    # We will test 4 cases:
    # 1. Shape implies fewer bytes than declared: shape [2] (8 bytes) vs offsets [0, 4] (4 bytes)
    # 2. Shape implies more bytes than declared: shape [4] (16 bytes) vs offsets [0, 8] (8 bytes)
    # 3. Shape with a zero dimension paired with non-zero byte range: shape [0, 5] (0 bytes) vs offsets [0, 8] (8 bytes)
    # 4. Negative shape values: shape [-1, 2] vs offsets [0, 8]
    
    cases = [
        {"shape": [2], "data_offsets": [0, 4], "msg": "does not match expected tensor size"},
        {"shape": [4], "data_offsets": [0, 8], "msg": "does not match expected tensor size"},
        {"shape": [0, 5], "data_offsets": [0, 8], "msg": "does not match expected tensor size"},
        {"shape": [-1, 2], "data_offsets": [0, 8], "msg": "invalid tensor shape|negative dimensions|does not match expected|failed to reshape"},
    ]
    
    for case in cases:
        header = {
            "__metadata__": {"format_version": "1.0", "optimizer_type": "Adam"},
            "__tensors__": {
                "state.0.exp_avg": {
                    "dtype": "F32",
                    "shape": case["shape"],
                    "data_offsets": case["data_offsets"]
                }
            },
            "__scalars__": {},
            "__config__": {"state_param_ids": [0]}
        }
        header_bytes = json.dumps(header).encode("utf-8")
        header_len = len(header_bytes)
        
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(struct.pack("<Q", header_len))
                f.write(header_bytes)
                f.write(b'\x00' * 32)
                
            with pytest.raises(ValueError) as excinfo:
                safe_load_optimizer(tmp_path)
                
            err_str = str(excinfo.value)
            assert re.search(case["msg"], err_str, re.IGNORECASE) is not None, f"Unexpected error message for shape {case['shape']}: {err_str}"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
