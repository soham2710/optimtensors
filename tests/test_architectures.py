import os
import tempfile
import pytest
import torch
import torch.nn as nn
from optimtensors.serde import safe_save_optimizer, safe_load_optimizer, safe_load_into_optimizer

# 1. CNNs
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 32 * 32, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 2. Sequence models
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 20, num_layers=2, batch_first=True)
        self.fc = nn.Linear(20, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
        
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(10, 20, num_layers=2, batch_first=True)
        self.fc = nn.Linear(20, 2)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# 3. Embedding-heavy models
class EmbeddingModel(nn.Module):
    def __init__(self, sparse=False):
        super().__init__()
        self.emb = nn.Embedding(20000, 128, sparse=sparse)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        h = self.emb(x).mean(dim=1)
        return self.fc(h)

# 4. Small/edge cases
class MinimalLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

class OneParamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return x * self.param


def run_architecture_test(model_name, model, optim_class, optim_kwargs, input_shape, is_embedding=False, target_shape=None, is_one_param=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = optim_class(model.parameters(), **optim_kwargs)
    
    # Run training step
    if is_embedding or "BERT" in model_name or "GPT" in model_name:
        vocab_size = 1000 if "GPT" in model_name else (30522 if "BERT" in model_name else 20000)
        x = torch.randint(0, vocab_size, input_shape, device=device)
        if target_shape:
            y = torch.randint(0, vocab_size if "GPT" in model_name else 2, target_shape, device=device)
        else:
            y = torch.randint(0, 2, (input_shape[0],), device=device)
    elif is_one_param:
        x = torch.randn(input_shape, device=device)
        y = torch.randn(input_shape, device=device)
    else:
        x = torch.randn(input_shape, device=device)
        if target_shape:
            y = torch.randint(0, 2, target_shape, device=device)
        else:
            y = torch.randint(0, 2, (input_shape[0],), device=device)
            
    optimizer.zero_grad()
    out = model(x)
    if hasattr(out, "logits"):
        out = out.logits
    if is_one_param:
        loss = nn.MSELoss()(out, y)
    elif target_shape:
        loss = nn.CrossEntropyLoss()(out.view(-1, out.size(-1)), y.view(-1))
    else:
        loss = nn.CrossEntropyLoss()(out, y)
        
    loss.backward()
    optimizer.step()
    
    orig_state_dict = optimizer.state_dict()
    
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Save
        safe_save_optimizer(orig_state_dict, tmp_path, optimizer_type=optim_class.__name__)
        
        # Load
        loaded_state_dict = safe_load_optimizer(tmp_path)
        
        # Verify keys
        assert set(orig_state_dict.keys()) == set(loaded_state_dict.keys())
        
        # Verify groups
        assert len(orig_state_dict["param_groups"]) == len(loaded_state_dict["param_groups"])
        
        # Verify states
        for p_id in orig_state_dict["state"]:
            orig_p = orig_state_dict["state"][p_id]
            loaded_p = loaded_state_dict["state"][p_id]
            assert set(orig_p.keys()) == set(loaded_p.keys())
            
            for k in orig_p:
                v_orig = orig_p[k]
                v_load = loaded_p[k]
                if isinstance(v_orig, torch.Tensor):
                    assert isinstance(v_load, torch.Tensor)
                    # Check on CPU
                    assert torch.equal(v_orig.cpu(), v_load)
                elif isinstance(v_orig, (list, tuple)) and len(v_orig) > 0 and any(isinstance(x, torch.Tensor) for x in v_orig):
                    assert isinstance(v_load, (list, tuple))
                    assert len(v_orig) == len(v_load)
                    for x_orig, x_load in zip(v_orig, v_load):
                        if x_orig is None:
                            assert x_load is None
                        else:
                            assert isinstance(x_load, torch.Tensor)
                            assert torch.equal(x_orig.cpu(), x_load)
                else:
                    assert v_orig == v_load
                    
        return "PASS"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"FAIL: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_diversity_matrix():
    # Import conditionally to avoid pytest failures if transformers/torchvision are still loading
    from transformers import BertConfig, BertForSequenceClassification, GPT2Config, GPT2LMHeadModel, ViTConfig, ViTForImageClassification
    try:
        import torchvision.models as torchvision_models
        HAS_TORCHVISION = True
    except ImportError:
        HAS_TORCHVISION = False
    
    matrix = []
    
    # 1. ResNet-18
    if HAS_TORCHVISION:
        resnet = torchvision_models.resnet18()
        for opt, kwargs in [(torch.optim.AdamW, {"lr": 1e-3}), (torch.optim.SGD, {"lr": 1e-2, "momentum": 0.9})]:
            status = run_architecture_test("ResNet-18", resnet, opt, kwargs, (2, 3, 224, 224))
            matrix.append(("ResNet-18", opt.__name__, status))
    else:
        matrix.append(("ResNet-18", "AdamW", "SKIPPED"))
        matrix.append(("ResNet-18", "SGD", "SKIPPED"))
        
    # 2. MobileNetV3-small
    if HAS_TORCHVISION:
        mobilenet = torchvision_models.mobilenet_v3_small()
        for opt, kwargs in [(torch.optim.Adam, {"lr": 1e-3}), (torch.optim.RMSprop, {"lr": 1e-3})]:
            status = run_architecture_test("MobileNetV3-small", mobilenet, opt, kwargs, (2, 3, 224, 224))
            matrix.append(("MobileNetV3-small", opt.__name__, status))
    else:
        matrix.append(("MobileNetV3-small", "Adam", "SKIPPED"))
        matrix.append(("MobileNetV3-small", "RMSprop", "SKIPPED"))
        
    # 3. SimpleConvNet
    convnet = SimpleConvNet()
    for opt, kwargs in [(torch.optim.AdamW, {"lr": 1e-3}), (torch.optim.SGD, {"lr": 1e-2, "momentum": 0.9})]:
        status = run_architecture_test("SimpleConvNet", convnet, opt, kwargs, (2, 3, 32, 32))
        matrix.append(("SimpleConvNet", opt.__name__, status))
        
    # 4. BERT-base (Small config)
    bert_config = BertConfig(hidden_size=128, num_hidden_layers=3, num_attention_heads=4, intermediate_size=256)
    bert_model = BertForSequenceClassification(bert_config)
    for opt, kwargs in [(torch.optim.AdamW, {"lr": 1e-3}), (torch.optim.Adam, {"lr": 1e-3})]:
        status = run_architecture_test("BERT-base-small", bert_model, opt, kwargs, (2, 16), target_shape=(2,))
        matrix.append(("BERT-base-small", opt.__name__, status))
        
    # 5. GPT-2-small (Small config)
    gpt2_config = GPT2Config(n_embd=128, n_layer=3, n_head=4, n_inner=256, vocab_size=1000)
    gpt2_model = GPT2LMHeadModel(gpt2_config)
    for opt, kwargs in [(torch.optim.AdamW, {"lr": 1e-3}), (torch.optim.RMSprop, {"lr": 1e-3})]:
        status = run_architecture_test("GPT-2-small", gpt2_model, opt, kwargs, (2, 16), target_shape=(2, 16))
        matrix.append(("GPT-2-small", opt.__name__, status))
        
    # 6. ViT-small (Small config)
    vit_config = ViTConfig(hidden_size=128, num_hidden_layers=3, num_attention_heads=4, intermediate_size=256, image_size=32, patch_size=8, num_channels=3)
    vit_model = ViTForImageClassification(vit_config)
    for opt, kwargs in [(torch.optim.AdamW, {"lr": 1e-3}), (torch.optim.SGD, {"lr": 1e-2, "momentum": 0.9})]:
        status = run_architecture_test("Vision-Transformer-small", vit_model, opt, kwargs, (2, 3, 32, 32))
        matrix.append(("Vision-Transformer-small", opt.__name__, status))
        
    # 7. LSTM
    lstm = LSTMModel()
    for opt, kwargs in [(torch.optim.Adam, {"lr": 1e-3}), (torch.optim.RMSprop, {"lr": 1e-3})]:
        status = run_architecture_test("LSTM", lstm, opt, kwargs, (2, 5, 10))
        matrix.append(("LSTM", opt.__name__, status))
        
    # 8. GRU
    gru = GRUModel()
    for opt, kwargs in [(torch.optim.AdamW, {"lr": 1e-3}), (torch.optim.SGD, {"lr": 1e-2, "momentum": 0.9})]:
        status = run_architecture_test("GRU", gru, opt, kwargs, (2, 5, 10))
        matrix.append(("GRU", opt.__name__, status))
        
    # 9. Embedding-heavy (Dense)
    emb_dense = EmbeddingModel(sparse=False)
    for opt, kwargs in [(torch.optim.AdamW, {"lr": 1e-3}), (torch.optim.SGD, {"lr": 1e-2})]:
        status = run_architecture_test("Embedding-Dense", emb_dense, opt, kwargs, (2, 10), is_embedding=True)
        matrix.append(("Embedding-Heavy-Dense", opt.__name__, status))
        
    # 10. Embedding-heavy (Sparse) - SGD supports sparse gradients
    emb_sparse = EmbeddingModel(sparse=True)
    for opt, kwargs in [(torch.optim.SGD, {"lr": 1e-2})]:
        status = run_architecture_test("Embedding-Sparse", emb_sparse, opt, kwargs, (2, 10), is_embedding=True)
        matrix.append(("Embedding-Heavy-Sparse", opt.__name__, status))
        
    # 11. Minimal Linear
    min_linear = MinimalLinear()
    for opt, kwargs in [(torch.optim.AdamW, {"lr": 1e-3}), (torch.optim.SGD, {"lr": 1e-2})]:
        status = run_architecture_test("nn.Linear(10,2)", min_linear, opt, kwargs, (2, 10))
        matrix.append(("nn.Linear(10,2)", opt.__name__, status))
        
    # 12. One Param Model
    one_param = OneParamModel()
    for opt, kwargs in [(torch.optim.Adam, {"lr": 1e-3}), (torch.optim.SGD, {"lr": 1e-2})]:
        status = run_architecture_test("One-Parameter", one_param, opt, kwargs, (2, 2), is_one_param=True)
        matrix.append(("One-Parameter", opt.__name__, status))
        
    # Append results table to test_results.md
    with open("test_results.md", "a") as f:
        f.write("\n## Section 3: Architectural Diversity Matrix Results\n\n")
        f.write("| Architecture | Optimizer | Status |\n")
        f.write("| :--- | :--- | :--- |\n")
        for row in matrix:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} |\n")
            
    # Verify all passed
    for row in matrix:
        if row[2] == "SKIPPED":
            continue
        assert row[2] == "PASS", f"Architecture test failed: {row[0]} with {row[1]}: {row[2]}"
