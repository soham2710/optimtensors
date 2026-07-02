import pytest
import torch
from optimtensors.type_check import validate_optimizer_state_dict, check_safe_structure


def test_allowed_types():
    # Valid primitives
    check_safe_structure(10)
    check_safe_structure(3.14)
    check_safe_structure(True)
    check_safe_structure("hello")
    check_safe_structure(None)
    
    # Valid lists & dicts
    check_safe_structure([1, 2, 3])
    check_safe_structure({"lr": 0.01, "betas": [0.9, 0.999]})
    check_safe_structure({"a": {"b": [True, False, None]}})


def test_disallowed_types():
    # Set is not allowed
    with pytest.raises(TypeError):
        check_safe_structure({1, 2, 3})
        
    # Function is not allowed
    with pytest.raises(TypeError):
        check_safe_structure(lambda x: x)
        
    # Custom class instance is not allowed
    class Dummy:
        pass
    with pytest.raises(TypeError):
        check_safe_structure(Dummy())
        
    # Dict with non-string/int keys is not allowed
    with pytest.raises(TypeError):
        check_safe_structure({(1, 2): "val"})
        
    # Lists with disallowed elements
    with pytest.raises(TypeError):
        check_safe_structure([1, lambda x: x, 3])


def test_optimizer_state_dict_validation():
    # Valid state dict
    valid_state = {
        "state": {
            0: {"step": 10, "exp_avg": torch.randn(2, 2)},
            1: {"step": 10, "exp_avg": torch.randn(3)}
        },
        "param_groups": [
            {"lr": 0.001, "betas": (0.9, 0.999), "params": [0, 1]}
        ]
    }
    validate_optimizer_state_dict(valid_state)
    
    # Invalid state key mapping (not a dict)
    invalid_state = {
        "state": [1, 2, 3],
        "param_groups": []
    }
    with pytest.raises(TypeError):
        validate_optimizer_state_dict(invalid_state)
        
    # Invalid type inside param state
    invalid_state_2 = {
        "state": {
            0: {"step": lambda x: x}
        },
        "param_groups": []
    }
    with pytest.raises(TypeError):
        validate_optimizer_state_dict(invalid_state_2)
        
    # Invalid key in state_dict (not list/dict but must be safe structure)
    invalid_state_3 = {
        "state": {},
        "param_groups": [],
        "custom_key": lambda x: x
    }
    with pytest.raises(TypeError):
        validate_optimizer_state_dict(invalid_state_3)


def test_quantized_optimizer_validation():
    class MockQuantState:
        pass
        
    quant_state_dict = {
        "state": {
            0: {
                "state1": torch.randint(0, 100, (2, 2), dtype=torch.int8),
                "quant_state1": MockQuantState()
            }
        },
        "param_groups": [{"params": [0], "lr": 0.001}]
    }
    
    with pytest.raises(ValueError) as excinfo:
        validate_optimizer_state_dict(quant_state_dict)
    assert "Unsupported optimizer state shape: 8-bit quantized optimizers" in str(excinfo.value)
