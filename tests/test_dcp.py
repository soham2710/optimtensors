import os
import json
import tempfile
import torch
import pytest

from optimtensors import SecureFileSystemWriter, SecureFileSystemReader
import torch.distributed.checkpoint as dcp

def test_secure_dcp_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create a dummy state dict
        orig_state_dict = {
            "weight_tensor": torch.randn(5, 5),
            "bias_tensor": torch.randn(5),
            "step_count": torch.tensor(42)
        }

        # 2. Save it using SecureFileSystemWriter
        writer = SecureFileSystemWriter(tmpdir)
        dcp.save(orig_state_dict, storage_writer=writer)

        # 3. Check that the metadata is stored as JSON and not pickle
        metadata_json_path = os.path.join(tmpdir, "metadata.json")
        metadata_pickle_path = os.path.join(tmpdir, ".metadata")
        
        assert os.path.exists(metadata_json_path), "metadata.json should exist"
        assert not os.path.exists(metadata_pickle_path), "Insecure pickle-based .metadata should NOT exist"

        # 4. Check that the metadata.json can be loaded as valid JSON
        with open(metadata_json_path, "r") as f:
            meta_content = json.load(f)
        
        assert "version" in meta_content
        assert "state_dict_metadata" in meta_content
        assert "storage_data" in meta_content
        
        # 5. Load it back using SecureFileSystemReader
        load_state_dict = {
            "weight_tensor": torch.empty(5, 5),
            "bias_tensor": torch.empty(5),
            "step_count": torch.empty((), dtype=torch.long)
        }
        
        reader = SecureFileSystemReader(tmpdir)
        dcp.load(load_state_dict, storage_reader=reader)

        # 6. Verify absolute equality
        assert torch.equal(orig_state_dict["weight_tensor"], load_state_dict["weight_tensor"])
        assert torch.equal(orig_state_dict["bias_tensor"], load_state_dict["bias_tensor"])
        assert torch.equal(orig_state_dict["step_count"], load_state_dict["step_count"])
