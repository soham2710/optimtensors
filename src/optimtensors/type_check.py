def check_safe_structure(val, path="", allow_tensors=False, allow_ndarrays=False):
    """
    Recursively validates that the structure only contains allowed safe primitive types:
    int, float, bool, str, None, and lists/tuples/dicts of these.
    Raises TypeError if any unsupported type is encountered.

    Note: integer dict keys are accepted, but JSON serialization coerces them
    to strings, so they round-trip through the file format as strings.
    """
    if isinstance(val, bool):
        return
    elif isinstance(val, (int, float, str)) or val is None:
        return
    elif allow_tensors and type(val).__name__ == "Tensor":
        return
    elif allow_ndarrays and (type(val).__name__ == "ndarray" or getattr(type(val), "__module__", "") == "numpy"):
        return
    elif isinstance(val, dict):
        for k, v in val.items():
            if not isinstance(k, (str, int)):
                raise TypeError(
                    f"Keys must be strings or integers, got {type(k).__name__} at {path}"
                )
            sub_path = f"{path}.{k}" if path else str(k)
            check_safe_structure(v, sub_path, allow_tensors, allow_ndarrays)
    elif isinstance(val, (list, tuple)):
        for i, item in enumerate(val):
            sub_path = f"{path}[{i}]"
            check_safe_structure(item, sub_path, allow_tensors, allow_ndarrays)
    else:
        allowed_msg = "Only int, float, bool, str, None, and lists/tuples/dicts of these are allowed."
        if allow_tensors or allow_ndarrays:
            allowed_msg = (
                "Only int, float, bool, str, None, lists/tuples/dicts of these, "
                f"and {'tensors' if allow_tensors else ''}{' and ' if allow_tensors and allow_ndarrays else ''}{'ndarrays' if allow_ndarrays else ''} are allowed."
            )
        raise TypeError(
            f"Unsupported type {type(val).__name__} at '{path}'. {allowed_msg}"
        )


def validate_optimizer_state_dict(state_dict: dict):
    """
    Validates the non-tensor parts of a PyTorch optimizer state_dict.
    It expects standard PyTorch optimizer keys ('state' and 'param_groups').
    """
    if not isinstance(state_dict, dict):
        raise TypeError("optimizer state_dict must be a dictionary")

    # Check for bitsandbytes 8-bit quantized optimizer structures
    state = state_dict.get("state", {})
    if isinstance(state, dict):
        for param_id, param_state in state.items():
            if isinstance(param_state, dict):
                for state_key, state_val in param_state.items():
                    if "quant_state" in state_key or "bitsandbytes" in str(type(state_val)):
                        raise ValueError(
                            f"Unsupported optimizer state shape: 8-bit quantized optimizers (e.g. from bitsandbytes) "
                            f"are not supported in safe-optim-checkpoint v1 due to custom non-tensor quantization states. "
                            f"Please use standard optimizers like 'adamw_torch'."
                        )

    # The top level keys should typically be 'state' and 'param_groups'.
    # If other keys exist, they must be safely structured.
    for k, v in state_dict.items():
        if k == 'state':
            if not isinstance(v, dict):
                raise TypeError("'state' key in state_dict must map to a dictionary")
            for param_id, param_state in v.items():
                if not isinstance(param_state, dict):
                    raise TypeError(f"State for parameter {param_id} must be a dictionary")
                for state_key, state_val in param_state.items():
                    # Tensors are processed separately, so they are allowed here.
                    # In python, we import torch inside or check type by name to avoid heavy dependency cycle,
                    # but since torch is imported in serde, we can check by type name or class here.
                    import torch
                    if isinstance(state_val, torch.Tensor):
                        continue
                    if isinstance(state_val, (list, tuple)) and len(state_val) > 0 and all(isinstance(x, (torch.Tensor, type(None))) for x in state_val) and any(isinstance(x, torch.Tensor) for x in state_val):
                        continue
                    check_safe_structure(state_val, f"state.{param_id}.{state_key}")
        elif k == 'param_groups':
            if not isinstance(v, list):
                raise TypeError("'param_groups' must be a list")
            for i, group in enumerate(v):
                if not isinstance(group, dict):
                    raise TypeError(f"param_groups[{i}] must be a dictionary")
                check_safe_structure(group, f"param_groups[{i}]")
        else:
            check_safe_structure(v, k)
