import json
import mmap
import os
import struct
import torch
from optimtensors.type_check import validate_optimizer_state_dict, check_safe_structure

TORCH_TO_SAFETA = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float64: "F64",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}

SAFETA_TO_TORCH = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

DTYPE_SIZES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float64: 8,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}


def get_scalar_type_name(val):
    if isinstance(val, bool):
        return "bool"
    elif isinstance(val, int):
        return "int"
    elif isinstance(val, float):
        return "float"
    elif isinstance(val, str):
        return "str"
    elif val is None:
        return "none"
    elif isinstance(val, (list, tuple)):
        return "list"
    else:
        raise TypeError(f"Unsupported scalar type: {type(val).__name__}")


def infer_optimizer_type(state_dict: dict) -> str:
    """Guesses the optimizer type based on the keys present in param_groups."""
    param_groups = state_dict.get("param_groups", [])
    if not param_groups:
        return "Unknown"
    
    first_group = param_groups[0]
    if "betas" in first_group:
        import warnings
        warnings.warn(
            "Optimizer type inferred as 'Adam' based on 'betas' in param_groups. "
            "Since Adam and AdamW share identical state_dict structures, "
            "this auto-inference is a best-effort fallback only. "
            "For accuracy, pass the 'optimizer_type' parameter explicitly to safe_save_optimizer.",
            UserWarning,
            stacklevel=2
        )
        return "Adam"
    elif "momentum" in first_group:
        return "SGD"
    elif "alpha" in first_group:
        return "RMSprop"
    elif "initial_accumulator_value" in first_group:
        return "Adagrad"
    elif "rho" in first_group:
        return "Adadelta"
    elif "max_iter" in first_group:
        return "LBFGS"
    return "Unknown"


def safe_save_optimizer(state_dict: dict, path: str, optimizer_type: str = None) -> None:
    """
    Saves a PyTorch optimizer state_dict to a file using the safe, zero-code-execution format.
    """
    # 1. Enforce strict type validation
    validate_optimizer_state_dict(state_dict)

    if optimizer_type is None:
        optimizer_type = infer_optimizer_type(state_dict)

    # 2. Deconstruct state_dict
    __metadata__ = {
        "format_version": "1.0",
        "optimizer_type": optimizer_type,
    }
    
    __tensors__ = {}
    __scalars__ = {}
    
    # Store parameter IDs and groups
    state_keys = list(state_dict.get("state", {}).keys())
    state_param_ids = [int(k) if isinstance(k, int) or (isinstance(k, str) and k.isdigit()) else k for k in state_keys]
    __config__ = {
        "param_groups": state_dict.get("param_groups", []),
        "state_param_ids": state_param_ids
    }

    # Extract tensors and scalars from 'state'
    tensors_to_write = []
    current_offset = 0

    state = state_dict.get("state", {})
    for param_id, param_state in state.items():
        param_id_str = str(param_id)
        for k, v in param_state.items():
            key_path = f"state.{param_id_str}.{k}"
            if isinstance(v, torch.Tensor):
                tensors_to_write.append((key_path, v))
                numel = v.numel()
                elem_size = v.element_size()
                total_bytes = numel * elem_size
                padding_len = (8 - (total_bytes % 8)) % 8
                
                dtype_str = TORCH_TO_SAFETA.get(v.dtype)
                if dtype_str is None:
                    raise TypeError(f"Unsupported tensor dtype: {v.dtype} at {key_path}")
                    
                __tensors__[key_path] = {
                    "dtype": dtype_str,
                    "shape": list(v.shape),
                    "data_offsets": [current_offset, current_offset + total_bytes]
                }
                current_offset += total_bytes + padding_len
                
            elif isinstance(v, (list, tuple)) and len(v) > 0 and any(isinstance(x, torch.Tensor) for x in v) and all(isinstance(x, (torch.Tensor, type(None))) for x in v):
                # Save as list of tensors (e.g. LBFGS state history)
                __scalars__[key_path] = {
                    "type": "tensor_list",
                    "value": len(v)
                }
                for idx, t in enumerate(v):
                    if t is not None:
                        sub_key = f"{key_path}.{idx}"
                        tensors_to_write.append((sub_key, t))
                        numel = t.numel()
                        elem_size = t.element_size()
                        total_bytes = numel * elem_size
                        padding_len = (8 - (total_bytes % 8)) % 8
                        
                        dtype_str = TORCH_TO_SAFETA.get(t.dtype)
                        if dtype_str is None:
                            raise TypeError(f"Unsupported tensor dtype: {t.dtype} at {sub_key}")
                            
                        __tensors__[sub_key] = {
                            "dtype": dtype_str,
                            "shape": list(t.shape),
                            "data_offsets": [current_offset, current_offset + total_bytes]
                        }
                        current_offset += total_bytes + padding_len
            else:
                # Store scalar with its type and value (tuples converted to list)
                val_to_store = list(v) if isinstance(v, tuple) else v
                __scalars__[key_path] = {
                    "type": get_scalar_type_name(v),
                    "value": val_to_store
                }

    # 3. Create JSON header
    header = {
        "__metadata__": __metadata__,
        "__tensors__": __tensors__,
        "__scalars__": __scalars__,
        "__config__": __config__
    }
    
    header_bytes = json.dumps(header).encode('utf-8')
    
    # 4. Align JSON header so the tensor buffer starts at an 8-byte boundary
    padding_len = (8 - (8 + len(header_bytes)) % 8) % 8
    header_bytes += b' ' * padding_len
    header_len = len(header_bytes)

    # 5. Write everything to disk
    # Ensure any parent directories exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        
        import ctypes
        for key_path, tensor in tensors_to_write:
            t_cpu = tensor.detach().cpu().contiguous()
            numel = t_cpu.numel()
            elem_size = t_cpu.element_size()
            total_bytes = numel * elem_size
            padding_len = (8 - (total_bytes % 8)) % 8
            
            if total_bytes > 0:
                address = t_cpu.untyped_storage().data_ptr()
                buffer = (ctypes.c_char * total_bytes).from_address(address)
                f.write(buffer)
                
            if padding_len > 0:
                f.write(b'\x00' * padding_len)


def safe_load_optimizer(path: str) -> dict:
    """
    Loads a PyTorch optimizer state_dict from a file securely, without code execution.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}")
        
    with open(path, "rb") as f:
        # Read header length
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Invalid file format: file too short to contain header length")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len > 50 * 1024 * 1024:
            raise ValueError(f"Header length ({header_len}) exceeds safety limit of 50MB")
        
        # Get file size to perform bounds-checking before allocation
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(8)
        
        if header_len + 8 > file_size:
            raise ValueError(f"Invalid header length {header_len} (file size is {file_size})")
            
        # Read JSON header
        header_bytes = f.read(header_len)
        if len(header_bytes) < header_len:
            raise ValueError("Invalid file: file truncated before header end")
        
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON header: {e}")
            
        # Verify required keys
        for key in ["__metadata__", "__tensors__", "__scalars__", "__config__"]:
            if key not in header:
                raise ValueError(f"Missing required top-level key: {key}")
                
        # Check for unexpected top-level keys to prevent any hidden slots
        allowed_keys = {"__metadata__", "__tensors__", "__scalars__", "__config__"}
        extra_keys = set(header.keys()) - allowed_keys
        if extra_keys:
            raise ValueError(f"Unexpected top-level keys in header: {extra_keys}")
            
        # Enforce strict types inside __scalars__ and __config__
        for k, v in header["__scalars__"].items():
            check_safe_structure(v, f"__scalars__.{k}")
        check_safe_structure(header["__config__"], "__config__")
        
        # Memory map the remaining raw buffer (using ACCESS_COPY for a writable buffer view)
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        
        start_offset = 8 + header_len
        tensor_data_len = file_size - start_offset
        
        if tensor_data_len > 0:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
        else:
            mm = b""

    try:
        # Reconstruct state_dict
        __tensors__ = header["__tensors__"]
        __scalars__ = header["__scalars__"]
        __config__ = header["__config__"]
        
        # Initialize state with empty dictionaries for all parameter IDs
        state_param_ids = __config__.get("state_param_ids", [])
        state = {param_id: {} for param_id in state_param_ids}
        
        # Reconstruct scalars FIRST (needed to pre-allocate tensor list placeholders)
        for key_path, scalar_entry in __scalars__.items():
            parts = key_path.split(".", 2)
            if len(parts) != 3 or parts[0] != "state":
                raise ValueError(f"Malformed scalar key in header: {key_path}")
                
            param_id_str = parts[1]
            param_id = int(param_id_str) if param_id_str.isdigit() else param_id_str
            state_key = parts[2]
            
            if scalar_entry["type"] == "tensor_list":
                state[param_id][state_key] = [None] * scalar_entry["value"]
            else:
                state[param_id][state_key] = scalar_entry["value"]

        # Reconstruct tensors
        for key_path, tensor_meta in __tensors__.items():
            # key_path is of the form: state.<param_id>.<state_key>
            parts = key_path.split(".", 2)
            if len(parts) != 3 or parts[0] != "state":
                raise ValueError(f"Malformed tensor key in header: {key_path}")
                
            param_id_str = parts[1]
            param_id = int(param_id_str) if param_id_str.isdigit() else param_id_str
            state_key = parts[2]
            
            dtype_str = tensor_meta["dtype"]
            shape = tensor_meta["shape"]
            start, end = tensor_meta["data_offsets"]
            
            torch_dtype = SAFETA_TO_TORCH.get(dtype_str)
            if torch_dtype is None:
                raise ValueError(f"Unsupported dtype in header: {dtype_str}")
                
            # Slice from mmap using absolute offset in file
            if start < 0 or end < start or start_offset + end > file_size:
                raise ValueError(f"Invalid tensor offsets [{start}, {end}] for key {key_path}")
            abs_start = start_offset + start
            abs_end = start_offset + end
            numel = 1
            for dim in shape:
                numel *= dim
                
            # Reconstruct tensor from buffer
            load_dtype = torch.int16 if torch_dtype == torch.bfloat16 else torch_dtype
            
            elem_size = DTYPE_SIZES.get(load_dtype, 1)
            if (end - start) != numel * elem_size:
                raise ValueError(
                    f"Offset range size ({end - start} bytes) does not match expected tensor size "
                    f"({numel} elements of size {elem_size} bytes = {numel * elem_size} bytes) for key {key_path}"
                )
            if numel > 0 and abs_start % elem_size != 0:
                raise ValueError(f"Unaligned memory offset {abs_start} for dtype {dtype_str} (element size {elem_size})")
                
            try:
                if numel == 0:
                    tensor = torch.empty(shape, dtype=load_dtype)
                else:
                    tensor = torch.frombuffer(mm, dtype=load_dtype, count=numel, offset=abs_start)
            except RuntimeError as e:
                raise ValueError(f"Failed to load tensor from buffer: {e}")
            
            if torch_dtype == torch.bfloat16:
                tensor = tensor.view(torch.bfloat16)
                
            tensor = tensor.reshape(shape)
            
            # Check if this tensor is part of a list of tensors
            if "." in state_key:
                base_key, index_str = state_key.rsplit(".", 1)
                if index_str.isdigit():
                    idx = int(index_str)
                    # Check that placeholder exists
                    if base_key not in state[param_id] or state[param_id][base_key] is None:
                        raise ValueError(f"Missing tensor list placeholder for {base_key}")
                    state[param_id][base_key][idx] = tensor
                else:
                    state[param_id][state_key] = tensor
            else:
                state[param_id][state_key] = tensor

        state_dict = {
            "state": state,
            "param_groups": __config__.get("param_groups", [])
        }
    except KeyError as e:
        raise ValueError(f"Malformed file header: missing expected key {e}")
    
    return state_dict


def safe_load_into_optimizer(optimizer: torch.optim.Optimizer, path: str) -> None:
    """
    Loads a PyTorch optimizer state_dict from a file securely and validates
    that all parameter shapes, dtypes, and counts match the target optimizer,
    preventing silent training issues.
    """
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("Expected argument 'optimizer' to be an instance of torch.optim.Optimizer")
        
    state_dict = safe_load_optimizer(path)
    
    # Validate structure
    orig_groups = state_dict.get("param_groups", [])
    loaded_state = state_dict.get("state", {})
    
    if len(orig_groups) != len(optimizer.param_groups):
        raise ValueError(
            f"Parameter group count mismatch: optimizer has {len(optimizer.param_groups)} groups, "
            f"but checkpoint has {len(orig_groups)} groups."
        )
        
    # Get all parameters in optimizer order
    all_params = []
    for group in optimizer.param_groups:
        all_params.extend(group["params"])
        
    # Validate each parameter state in loaded_state
    for param_id_str, param_state in loaded_state.items():
        param_id = int(param_id_str)
        if param_id < 0 or param_id >= len(all_params):
            raise ValueError(
                f"Checkpoint state refers to parameter index {param_id}, "
                f"but optimizer only has {len(all_params)} parameters."
            )
            
        p = all_params[param_id]
        
        # Check shapes and dtypes
        for state_key, val in param_state.items():
            # If the parameter state tensor is already populated in optimizer, check against it.
            # Otherwise, check against parameter p if the sizes match.
            if p in optimizer.state and state_key in optimizer.state[p]:
                expected_val = optimizer.state[p][state_key]
                if isinstance(expected_val, torch.Tensor):
                    if not isinstance(val, torch.Tensor):
                        raise ValueError(
                            f"Type mismatch for state '{state_key}' of parameter {param_id}: "
                            f"expected Tensor, got {type(val).__name__}"
                        )
                    if val.shape != expected_val.shape:
                        raise ValueError(
                            f"Shape mismatch for state '{state_key}' of parameter {param_id}: "
                            f"expected {expected_val.shape}, got {val.shape}"
                        )
                    if val.dtype != expected_val.dtype:
                        raise ValueError(
                            f"Dtype mismatch for state '{state_key}' of parameter {param_id}: "
                            f"expected {expected_val.dtype}, got {val.dtype}"
                        )
            else:
                # If state is not initialized yet in optimizer, check if it's a tensor.
                # If it's a tensor of size > 1, we expect it to match parameter shape p.shape.
                if isinstance(val, torch.Tensor):
                    if val.dim() == 0:
                        continue
                    if val.shape != p.shape:
                        if val.numel() > 1 or p.numel() == 1:
                            raise ValueError(
                                f"Shape mismatch for state '{state_key}' of parameter {param_id}: "
                                f"expected {p.shape}, got {val.shape}"
                            )
                elif isinstance(val, (list, tuple)) and len(val) > 0 and any(isinstance(x, torch.Tensor) for x in val):
                    # For list/tuple of tensors (e.g. LBFGS state history), check elements
                    for idx, item in enumerate(val):
                        if isinstance(item, torch.Tensor):
                            if item.dim() == 0:
                                continue
                            if item.shape != p.shape:
                                if item.numel() > 1 or p.numel() == 1:
                                    raise ValueError(
                                        f"Shape mismatch for state '{state_key}[{idx}]' of parameter {param_id}: "
                                        f"expected {p.shape}, got {item.shape}"
                                    )

    # Load the validated state_dict into the optimizer
    optimizer.load_state_dict(state_dict)
