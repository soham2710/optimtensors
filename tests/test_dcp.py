import os
import json
import tempfile
import torch
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from hypothesis import given, settings, strategies as st

from optimtensors import SecureFileSystemWriter, SecureFileSystemReader
from optimtensors.dcp import (
    metadata_to_json_dict,
    json_dict_to_metadata,
    _dict_to_tensor_properties,
    _tensor_properties_to_dict
)
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import (
    Metadata,
    TensorStorageMetadata,
    BytesStorageMetadata,
    TensorProperties,
    ChunkStorageMetadata,
    StorageMeta,
    MetadataIndex
)
from torch.distributed.checkpoint.filesystem import _StorageInfo

# 1. Basic secure DCP saving/loading roundtrip
def test_secure_dcp_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy state dict
        orig_state_dict = {
            "weight_tensor": torch.randn(5, 5),
            "bias_tensor": torch.randn(5),
            "step_count": torch.tensor(42)
        }

        # Save it using SecureFileSystemWriter
        writer = SecureFileSystemWriter(tmpdir)
        dcp.save(orig_state_dict, storage_writer=writer)

        # Check that the metadata is stored as JSON and not pickle
        metadata_json_path = os.path.join(tmpdir, "metadata.json")
        metadata_pickle_path = os.path.join(tmpdir, ".metadata")
        
        assert os.path.exists(metadata_json_path), "metadata.json should exist"
        assert not os.path.exists(metadata_pickle_path), "Insecure pickle-based .metadata should NOT exist"

        # Check that the metadata.json can be loaded as valid JSON
        with open(metadata_json_path, "r") as f:
            meta_content = json.load(f)
        
        assert "version" in meta_content
        assert "state_dict_metadata" in meta_content
        assert "storage_data" in meta_content
        
        # Load it back using SecureFileSystemReader
        load_state_dict = {
            "weight_tensor": torch.empty(5, 5),
            "bias_tensor": torch.empty(5),
            "step_count": torch.empty((), dtype=torch.long)
        }
        
        reader = SecureFileSystemReader(tmpdir)
        dcp.load(load_state_dict, storage_reader=reader)

        # Verify absolute equality
        assert torch.equal(orig_state_dict["weight_tensor"], load_state_dict["weight_tensor"])
        assert torch.equal(orig_state_dict["bias_tensor"], load_state_dict["bias_tensor"])
        assert torch.equal(orig_state_dict["step_count"], load_state_dict["step_count"])

# 2. Multi-datatype Roundtrip test
def test_secure_dcp_multitype():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_state_dict = {
            "float32_t": torch.randn(2, 3, dtype=torch.float32),
            "float16_t": torch.randn(4, dtype=torch.float16),
            "bfloat16_t": torch.randn(3, dtype=torch.bfloat16),
            "float64_t": torch.randn(2, dtype=torch.float64),
            "int32_t": torch.tensor([1, -2, 3], dtype=torch.int32),
            "int64_t": torch.tensor([1000, 2000], dtype=torch.int64),
            "bool_t": torch.tensor([True, False, True], dtype=torch.bool)
        }

        writer = SecureFileSystemWriter(tmpdir)
        dcp.save(orig_state_dict, storage_writer=writer)

        load_state_dict = {
            "float32_t": torch.empty(2, 3, dtype=torch.float32),
            "float16_t": torch.empty(4, dtype=torch.float16),
            "bfloat16_t": torch.empty(3, dtype=torch.bfloat16),
            "float64_t": torch.empty(2, dtype=torch.float64),
            "int32_t": torch.empty(3, dtype=torch.int32),
            "int64_t": torch.empty(2, dtype=torch.int64),
            "bool_t": torch.empty(3, dtype=torch.bool)
        }

        reader = SecureFileSystemReader(tmpdir)
        dcp.load(load_state_dict, storage_reader=reader)

        for k, v in orig_state_dict.items():
            assert load_state_dict[k].dtype == v.dtype, f"Dtype mismatch for {k}"
            assert torch.equal(v, load_state_dict[k]), f"Value mismatch for {k}"

# 3. Hypothesis Property-Based Testing for Metadata serialization
dtypes_strategy = st.sampled_from([
    torch.float32, torch.float16, torch.bfloat16, torch.float64,
    torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool
])

layouts_strategy = st.sampled_from([torch.strided])
mem_formats_strategy = st.sampled_from([
    torch.contiguous_format, torch.channels_last, torch.preserve_format
])

@st.composite
def tensor_properties_strategy(draw):
    return TensorProperties(
        dtype=draw(dtypes_strategy),
        layout=draw(layouts_strategy),
        requires_grad=draw(st.booleans()),
        memory_format=draw(mem_formats_strategy),
        pin_memory=draw(st.booleans())
    )

@st.composite
def chunk_metadata_strategy(draw):
    size_len = draw(st.integers(0, 4))
    offsets = [draw(st.integers(0, 100)) for _ in range(size_len)]
    sizes = [draw(st.integers(1, 1000)) for _ in range(size_len)]
    return ChunkStorageMetadata(
        offsets=torch.Size(offsets),
        sizes=torch.Size(sizes)
    )

@st.composite
def tensor_storage_metadata_strategy(draw):
    props = draw(tensor_properties_strategy())
    size_len = draw(st.integers(0, 4))
    size = torch.Size([draw(st.integers(1, 100)) for _ in range(size_len)])
    chunks_len = draw(st.integers(0, 5))
    chunks = [draw(chunk_metadata_strategy()) for _ in range(chunks_len)]
    return TensorStorageMetadata(properties=props, size=size, chunks=chunks)

@st.composite
def storage_info_strategy(draw):
    return _StorageInfo(
        relative_path=draw(st.text(min_size=1, max_size=50)),
        offset=draw(st.integers(0, 1000000)),
        length=draw(st.integers(1, 10000000)),
        transform_descriptors=draw(st.lists(st.text(min_size=1, max_size=20), max_size=3))
    )

@st.composite
def metadata_index_strategy(draw):
    fqn = draw(st.text(min_size=1, max_size=50))
    has_offset = draw(st.booleans())
    offset = None
    if has_offset:
        offset_len = draw(st.integers(0, 4))
        offset = torch.Size([draw(st.integers(0, 1000)) for _ in range(offset_len)])
    index = draw(st.one_of(st.none(), st.integers(0, 100)))
    return MetadataIndex(fqn=fqn, offset=offset, index=index)

@st.composite
def storage_meta_strategy(draw):
    return StorageMeta(
        checkpoint_id=draw(st.text(min_size=1, max_size=100)),
        save_id=draw(st.text(min_size=1, max_size=100)),
        load_id=draw(st.text(min_size=1, max_size=100)),
        modules=draw(st.lists(st.text(min_size=1, max_size=50), max_size=5))
    )

@st.composite
def metadata_strategy(draw):
    num_tensors = draw(st.integers(0, 10))
    state_dict_metadata = {}
    for i in range(num_tensors):
        fqn = f"tensor_{i}"
        is_bytes = draw(st.booleans())
        if is_bytes:
            state_dict_metadata[fqn] = BytesStorageMetadata()
        else:
            state_dict_metadata[fqn] = draw(tensor_storage_metadata_strategy())
            
    num_storage_info = draw(st.integers(0, 5))
    storage_data = {}
    for _ in range(num_storage_info):
        idx = draw(metadata_index_strategy())
        info = draw(storage_info_strategy())
        storage_data[idx] = info
        
    storage_meta = draw(st.one_of(st.none(), storage_meta_strategy()))
    version = draw(st.text(min_size=1, max_size=10))
    
    return Metadata(
        state_dict_metadata=state_dict_metadata,
        planner_data={"some_key": "some_value"},
        storage_data=storage_data if storage_data else None,
        storage_meta=storage_meta,
        version=version
    )

@settings(max_examples=50)
@given(metadata_strategy())
def test_metadata_serialization_roundtrip(meta):
    json_dict = metadata_to_json_dict(meta)
    
    # Check that it serializes to string and parses back nicely
    serialized = json.dumps(json_dict)
    deserialized_json = json.loads(serialized)
    
    rebuilt_meta = json_dict_to_metadata(deserialized_json)
    
    # Assert correctness
    assert rebuilt_meta.version == meta.version
    assert rebuilt_meta.planner_data == meta.planner_data
    
    # Assert state_dict_metadata structures match
    assert len(rebuilt_meta.state_dict_metadata) == len(meta.state_dict_metadata)
    for k, v in meta.state_dict_metadata.items():
        rebuilt_v = rebuilt_meta.state_dict_metadata[k]
        assert type(rebuilt_v) is type(v)
        if isinstance(v, TensorStorageMetadata):
            assert rebuilt_v.size == v.size
            assert len(rebuilt_v.chunks) == len(v.chunks)
            for c1, c2 in zip(rebuilt_v.chunks, v.chunks):
                assert c1.offsets == c2.offsets
                assert c1.sizes == c2.sizes
            # Properties
            assert rebuilt_v.properties.dtype == v.properties.dtype
            assert rebuilt_v.properties.requires_grad == v.properties.requires_grad
            assert rebuilt_v.properties.pin_memory == v.properties.pin_memory
            
    # Assert storage_data matches
    if meta.storage_data is None:
        assert rebuilt_meta.storage_data is None
    else:
        assert len(rebuilt_meta.storage_data) == len(meta.storage_data)
        for idx, info in meta.storage_data.items():
            # Find matching idx in rebuilt_meta
            match_idx = None
            for r_idx in rebuilt_meta.storage_data.keys():
                if r_idx.fqn == idx.fqn and r_idx.offset == idx.offset and r_idx.index == idx.index:
                    match_idx = r_idx
                    break
            assert match_idx is not None, f"Could not find matching metadata index for FQN: {idx.fqn}"
            rebuilt_info = rebuilt_meta.storage_data[match_idx]
            assert rebuilt_info.relative_path == info.relative_path
            assert rebuilt_info.offset == info.offset
            assert rebuilt_info.length == info.length
            assert rebuilt_info.transform_descriptors == info.transform_descriptors

    # Assert storage_meta matches
    if meta.storage_meta is None:
        assert rebuilt_meta.storage_meta is None
    else:
        assert rebuilt_meta.storage_meta.checkpoint_id == meta.storage_meta.checkpoint_id
        assert rebuilt_meta.storage_meta.save_id == meta.storage_meta.save_id
        assert rebuilt_meta.storage_meta.load_id == meta.storage_meta.load_id
        assert rebuilt_meta.storage_meta.modules == meta.storage_meta.modules

# 4. Fuzz / Error Handling tests
def test_secure_dcp_corrupted_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write corrupted JSON metadata
        metadata_json_path = os.path.join(tmpdir, "metadata.json")
        with open(metadata_json_path, "w") as f:
            f.write("{invalid json: [")
            
        reader = SecureFileSystemReader(tmpdir)
        load_state_dict = {"bias_tensor": torch.empty(5)}
        
        with pytest.raises(BaseException) as excinfo:
            dcp.load(load_state_dict, storage_reader=reader)
            
        assert "double quotes" in str(excinfo.value) or "JSON" in str(excinfo.value)

# 5. Distributed Multiprocess Gloo Test (CPU Simulation)
def _run_distributed_save_load(rank, world_size, temp_dir, sync_file):
    # Initialize the process group with gloo backend on CPU
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{sync_file}",
        world_size=world_size,
        rank=rank
    )
    
    # Setup dummy rank-specific tensor data
    my_state_dict = {
        f"rank_tensor_{rank}": torch.ones(5, 5, dtype=torch.float32) * (rank + 1)
    }
    
    # Save the checkpoint distributedly using our custom SecureFileSystemWriter
    writer = SecureFileSystemWriter(temp_dir)
    dcp.save(my_state_dict, storage_writer=writer)
    
    # Barrier synchronization
    dist.barrier()
    
    # Load the checkpoint back into fresh tensors
    load_state_dict = {
        f"rank_tensor_{rank}": torch.empty(5, 5, dtype=torch.float32)
    }
    
    reader = SecureFileSystemReader(temp_dir)
    dcp.load(load_state_dict, storage_reader=reader)
    
    # Verify correctness
    expected = torch.ones(5, 5, dtype=torch.float32) * (rank + 1)
    assert torch.equal(load_state_dict[f"rank_tensor_{rank}"], expected), f"Rank {rank} loaded incorrect tensor values"
    
    # Clean up process group
    dist.destroy_process_group()

def test_secure_dcp_distributed_gloo():
    # Setup temporary directories and synchronization files
    with tempfile.TemporaryDirectory() as tmpdir:
        sync_fd, sync_file = tempfile.mkstemp()
        os.close(sync_fd)
        
        world_size = 2
        try:
            mp.spawn(
                _run_distributed_save_load,
                args=(world_size, tmpdir, sync_file),
                nprocs=world_size,
                join=True
            )
            
            # Verify metadata file is JSON and contains expected rank-specific info
            metadata_json_path = os.path.join(tmpdir, "metadata.json")
            assert os.path.exists(metadata_json_path), "metadata.json should be created by finish()"
            
            with open(metadata_json_path, "r") as f:
                meta_content = json.load(f)
                
            # Verify metadata properties
            assert "state_dict_metadata" in meta_content
            # The metadata should contain keys for both rank tensors
            assert "rank_tensor_0" in meta_content["state_dict_metadata"]
            assert "rank_tensor_1" in meta_content["state_dict_metadata"]
            
        finally:
            if os.path.exists(sync_file):
                os.remove(sync_file)

