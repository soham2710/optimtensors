from optimtensors.serde import safe_save_optimizer, safe_load_optimizer, safe_load_into_optimizer
from optimtensors.integrations import OptimTensorsCallback, OptimTensorsTrainerMixin, load_trainer_optimizer
from optimtensors.distributed import (
    save_fsdp_full_optimizer,
    load_fsdp_full_optimizer,
    save_sharded_optimizer,
    load_sharded_optimizer,
)

__all__ = [
    "safe_save_optimizer",
    "safe_load_optimizer",
    "safe_load_into_optimizer",
    "OptimTensorsCallback",
    "OptimTensorsTrainerMixin",
    "load_trainer_optimizer",
    "save_fsdp_full_optimizer",
    "load_fsdp_full_optimizer",
    "save_sharded_optimizer",
    "load_sharded_optimizer",
]

# The DCP integration depends on recent torch.distributed.checkpoint
# internals (CURRENT_DCP_VERSION, _StorageInfo.transform_descriptors, ...).
# Keep the core package importable on older torch versions.
try:
    from optimtensors.dcp import SecureFileSystemWriter, SecureFileSystemReader
    __all__ += ["SecureFileSystemWriter", "SecureFileSystemReader"]
    DCP_AVAILABLE = True
except ImportError:
    DCP_AVAILABLE = False
