import os
import json
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.distributed.checkpoint as dcp

from optimtensors import SecureFileSystemWriter, SecureFileSystemReader
from test_architectures import SimpleConvNet, EmbeddingModel, LSTMModel

# Conditionally import transformers/torchvision
try:
    from transformers import BertConfig, BertForSequenceClassification, GPT2Config, GPT2LMHeadModel, ViTConfig, ViTForImageClassification
    import torchvision.models as torchvision_models
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

def run_dcp_architecture_test(name, model, optimizer_cls, optimizer_kwargs, input_shape_or_inputs):
    # Determine devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # 1. Initialize optimizer and populate states with a dummy step
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    
    if isinstance(input_shape_or_inputs, tuple):
        inputs = torch.randn(*input_shape_or_inputs, device=device)
    else:
        inputs = input_shape_or_inputs
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)
            
    # Forward pass
    optimizer.zero_grad()
    if isinstance(inputs, dict):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0].sum()
    else:
        outputs = model(inputs)
        loss = outputs.sum()
        
    loss.backward()
    optimizer.step()
    
    # Get original state dict
    orig_state_dict = optimizer.state_dict()
    
    # 2. Save distributedly using SecureFileSystemWriter
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SecureFileSystemWriter(tmpdir)
        dcp.save(orig_state_dict, storage_writer=writer)
        
        # Verify metadata
        metadata_json_path = os.path.join(tmpdir, "metadata.json")
        assert os.path.exists(metadata_json_path), f"[{name}] metadata.json was not created"
        
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)
            
        assert "state_dict_metadata" in metadata, f"[{name}] state_dict_metadata missing from JSON"
        
        # 3. Load distributedly using SecureFileSystemReader into a fresh optimizer state
        fresh_optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
        fresh_state_dict = fresh_optimizer.state_dict()
        
        # Create empty placeholder tensors matching the shapes in orig_state_dict
        for param_id, state in orig_state_dict["state"].items():
            fresh_state_dict["state"][param_id] = {}
            for state_key, state_val in state.items():
                if isinstance(state_val, torch.Tensor):
                    fresh_state_dict["state"][param_id][state_key] = torch.empty_like(state_val)
                else:
                    fresh_state_dict["state"][param_id][state_key] = state_val
                    
        reader = SecureFileSystemReader(tmpdir)
        dcp.load(fresh_state_dict, storage_reader=reader)
        
        # 4. Verify bitwise equivalence
        for param_id, state in orig_state_dict["state"].items():
            for state_key, state_val in state.items():
                if isinstance(state_val, torch.Tensor):
                    loaded_val = fresh_state_dict["state"][param_id][state_key]
                    assert torch.equal(state_val.cpu(), loaded_val.cpu()), \
                        f"[{name}] Optimizer state '{state_key}' mismatch for param {param_id}"

@pytest.mark.skipif(not HAS_DEPENDENCIES, reason="Missing torchvision or transformers")
def test_dcp_architecture_diversity_matrix():
    # 1. ResNet-18 (CNN) + AdamW
    resnet = torchvision_models.resnet18()
    run_dcp_architecture_test(
        "ResNet-18 + AdamW",
        resnet,
        torch.optim.AdamW,
        {"lr": 1e-3},
        (2, 3, 224, 224)
    )
    
    # 2. MobileNetV3-small + Adam
    mobilenet = torchvision_models.mobilenet_v3_small()
    run_dcp_architecture_test(
        "MobileNetV3 + Adam",
        mobilenet,
        torch.optim.Adam,
        {"lr": 1e-3},
        (2, 3, 224, 224)
    )
    
    # 3. SimpleConvNet + SGD
    convnet = SimpleConvNet()
    run_dcp_architecture_test(
        "SimpleConvNet + SGD",
        convnet,
        torch.optim.SGD,
        {"lr": 1e-2, "momentum": 0.9},
        (2, 3, 32, 32)
    )

    # 4. BERT (Transformer Encoder) + AdamW
    bert_config = BertConfig(
        vocab_size=1000, hidden_size=128, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=256
    )
    bert_model = BertForSequenceClassification(bert_config)
    dummy_input_ids = torch.randint(0, 1000, (2, 32))
    dummy_labels = torch.randint(0, 2, (2,))
    run_dcp_architecture_test(
        "BERT + AdamW",
        bert_model,
        torch.optim.AdamW,
        {"lr": 1e-4},
        {"input_ids": dummy_input_ids, "labels": dummy_labels}
    )

    # 5. GPT-2 (Transformer Decoder) + RMSprop
    gpt_config = GPT2Config(
        vocab_size=1000, n_embd=128, n_layer=2, n_head=2, n_inner=256
    )
    gpt_model = GPT2LMHeadModel(gpt_config)
    dummy_gpt_input_ids = torch.randint(0, 1000, (2, 16))
    run_dcp_architecture_test(
        "GPT-2 + RMSprop",
        gpt_model,
        torch.optim.RMSprop,
        {"lr": 1e-4},
        {"input_ids": dummy_gpt_input_ids, "labels": dummy_gpt_input_ids}
    )

    # 6. Vision Transformer (ViT) + SGD
    vit_config = ViTConfig(
        image_size=32, patch_size=4, num_channels=3,
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=128
    )
    vit_model = ViTForImageClassification(vit_config)
    dummy_pixel_values = torch.randn(2, 3, 32, 32)
    dummy_labels = torch.randint(0, 2, (2,))
    run_dcp_architecture_test(
        "ViT + SGD",
        vit_model,
        torch.optim.SGD,
        {"lr": 1e-3},
        {"pixel_values": dummy_pixel_values, "labels": dummy_labels}
    )

    # 7. LSTM + AdamW
    lstm_model = LSTMModel()
    dummy_lstm_input = torch.randn(4, 3, 10) # sequence_length, batch_size, input_size
    run_dcp_architecture_test(
        "LSTM + AdamW",
        lstm_model,
        torch.optim.AdamW,
        {"lr": 1e-3},
        dummy_lstm_input
    )

    # 8. Embedding-Heavy + Adagrad
    embed_model = EmbeddingModel(sparse=False)
    dummy_ids = torch.randint(0, 100, (4, 10))
    run_dcp_architecture_test(
        "EmbeddingHeavy + Adagrad",
        embed_model,
        torch.optim.Adagrad,
        {"lr": 1e-2},
        dummy_ids
    )
