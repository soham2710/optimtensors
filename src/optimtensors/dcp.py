import os
import json
from typing import Any, cast
from pathlib import Path

import torch
from torch.distributed.checkpoint.metadata import (
    Metadata,
    TensorStorageMetadata,
    BytesStorageMetadata,
    TensorProperties,
    ChunkStorageMetadata,
    StorageMeta,
    MetadataIndex
)
from torch.distributed.checkpoint.filesystem import (
    FileSystemWriter,
    FileSystemReader,
    _StorageInfo,
    CURRENT_DCP_VERSION
)
from torch.distributed.checkpoint.storage import WriteResult

def _tensor_properties_to_dict(props: TensorProperties) -> dict:
    return {
        "dtype": str(props.dtype),
        "requires_grad": props.requires_grad,
        "pin_memory": props.pin_memory,
        "layout": str(props.layout),
        "memory_format": str(props.memory_format),
    }

def _dict_to_tensor_properties(d: dict) -> TensorProperties:
    dtype_str = d.get("dtype", "torch.float32")
    if dtype_str.startswith("torch."):
        dtype = getattr(torch, dtype_str.split(".")[1])
    else:
        dtype = torch.float32
        
    layout_str = d.get("layout", "torch.strided")
    if layout_str.startswith("torch."):
        layout = getattr(torch, layout_str.split(".")[1])
    else:
        layout = torch.strided

    mem_format_str = d.get("memory_format", "torch.contiguous_format")
    if "contiguous_format" in mem_format_str:
        memory_format = torch.contiguous_format
    elif "channels_last" in mem_format_str:
        memory_format = torch.channels_last
    elif "preserve_format" in mem_format_str:
        memory_format = torch.preserve_format
    else:
        memory_format = torch.contiguous_format

    return TensorProperties(
        dtype=dtype,
        layout=layout,
        requires_grad=d.get("requires_grad", False),
        memory_format=memory_format,
        pin_memory=d.get("pin_memory", False)
    )

def _chunk_metadata_to_dict(chunk: ChunkStorageMetadata) -> dict:
    return {
        "offsets": list(chunk.offsets),
        "sizes": list(chunk.sizes)
    }

def _dict_to_chunk_metadata(d: dict) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size(d["offsets"]),
        sizes=torch.Size(d["sizes"])
    )

def _storage_type_to_dict(val: Any) -> dict:
    if isinstance(val, TensorStorageMetadata):
        return {
            "type": "TensorStorageMetadata",
            "properties": _tensor_properties_to_dict(val.properties),
            "size": list(val.size),
            "chunks": [_chunk_metadata_to_dict(c) for c in val.chunks]
        }
    elif isinstance(val, BytesStorageMetadata):
        return {
            "type": "BytesStorageMetadata"
        }
    else:
        raise ValueError(f"Unknown storage type: {val}")

def _dict_to_storage_type(d: dict) -> Any:
    t = d["type"]
    if t == "TensorStorageMetadata":
        return TensorStorageMetadata(
            properties=_dict_to_tensor_properties(d["properties"]),
            size=torch.Size(d["size"]),
            chunks=[_dict_to_chunk_metadata(c) for c in d["chunks"]]
        )
    elif t == "BytesStorageMetadata":
        return BytesStorageMetadata()
    else:
        raise ValueError(f"Unknown storage type: {t}")

def _metadata_index_to_dict(idx: MetadataIndex) -> dict:
    return {
        "fqn": idx.fqn,
        "offset": list(idx.offset) if idx.offset is not None else None,
        "index": idx.index
    }

def _dict_to_metadata_index(d: dict) -> MetadataIndex:
    return MetadataIndex(
        fqn=d["fqn"],
        offset=d["offset"],
        index=d["index"]
    )

def _storage_info_to_dict(info: _StorageInfo) -> dict:
    return {
        "relative_path": info.relative_path,
        "offset": info.offset,
        "length": info.length,
        "transform_descriptors": list(info.transform_descriptors) if info.transform_descriptors is not None else None
    }

def _dict_to_storage_info(d: dict) -> _StorageInfo:
    return _StorageInfo(
        relative_path=d["relative_path"],
        offset=d["offset"],
        length=d["length"],
        transform_descriptors=d["transform_descriptors"]
    )

def _storage_meta_to_dict(meta: StorageMeta) -> dict:
    if meta is None:
        return None
    return {
        "checkpoint_id": str(meta.checkpoint_id) if meta.checkpoint_id is not None else None,
        "save_id": meta.save_id,
        "load_id": meta.load_id,
        "modules": meta.modules
    }

def _dict_to_storage_meta(d: dict) -> StorageMeta:
    if d is None:
        return None
    return StorageMeta(
        checkpoint_id=d["checkpoint_id"],
        save_id=d["save_id"],
        load_id=d["load_id"],
        modules=d["modules"]
    )

def metadata_to_json_dict(metadata: Metadata) -> dict:
    state_dict_meta_dict = {
        k: _storage_type_to_dict(v) for k, v in metadata.state_dict_metadata.items()
    }
    
    storage_data_list = []
    if metadata.storage_data is not None:
        for idx, info in metadata.storage_data.items():
            storage_data_list.append({
                "index": _metadata_index_to_dict(idx),
                "info": _storage_info_to_dict(info)
            })
            
    return {
        "version": metadata.version,
        "state_dict_metadata": state_dict_meta_dict,
        "planner_data": metadata.planner_data,
        "storage_data": storage_data_list,
        "storage_meta": _storage_meta_to_dict(metadata.storage_meta)
    }

def json_dict_to_metadata(d: dict) -> Metadata:
    state_dict_metadata = {
        k: _dict_to_storage_type(v) for k, v in d["state_dict_metadata"].items()
    }
    
    storage_data = {}
    if d.get("storage_data") is not None:
        for item in d["storage_data"]:
            idx = _dict_to_metadata_index(item["index"])
            info = _dict_to_storage_info(item["info"])
            storage_data[idx] = info
            
    return Metadata(
        state_dict_metadata=state_dict_metadata,
        planner_data=d.get("planner_data"),
        storage_data=storage_data if storage_data else None,
        storage_meta=_dict_to_storage_meta(d.get("storage_meta")),
        version=d.get("version")
    )

class SecureFileSystemWriter(FileSystemWriter):
    """
    Subclass of FileSystemWriter that serializes the checkpoint metadata
    in a secure, human-readable JSON format instead of standard Python pickle.
    """
    def _get_metadata_path(self, rank: int | None = None) -> os.PathLike:
        filename = "metadata.json" if rank is None else f"__{rank}_metadata.json"
        return cast(Path, self.fs.concat_path(self.path, filename))

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        metadata.version = CURRENT_DCP_VERSION

        storage_md = {}
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md

        metadata.storage_meta = self.storage_meta()
        tmp_filename = (
            f"__{self.rank}_metadata.json.tmp"
            if not self.use_collectives and self.rank is not None
            else "metadata.json.tmp"
        )
        tmp_path = cast(Path, self.fs.concat_path(self.path, tmp_filename))
        
        json_dict = metadata_to_json_dict(metadata)
        with self.fs.create_stream(tmp_path, "w") as metadata_file:
            json.dump(json_dict, metadata_file, indent=2)
            
        dest_filename = (
            f"__{self.rank}_metadata.json"
            if not self.use_collectives and self.rank is not None
            else "metadata.json"
        )
        dest_path = cast(Path, self.fs.concat_path(self.path, dest_filename))
        self.fs.rename(tmp_path, dest_path)

class SecureFileSystemReader(FileSystemReader):
    """
    Subclass of FileSystemReader that deserializes the checkpoint metadata
    from a secure, human-readable JSON format instead of standard Python pickle.
    """
    def _get_metadata_path(self, rank: int | None = None) -> os.PathLike:
        filename = "metadata.json" if rank is None else f"__{rank}_metadata.json"
        return cast(Path, self.fs.concat_path(self.path, filename))

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        rank = kwargs.get("rank")
        path = self._get_metadata_path(rank)
        with self.fs.create_stream(path, "r") as metadata_file:
            json_dict = json.load(metadata_file)
        metadata = json_dict_to_metadata(json_dict)

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id

        return metadata
