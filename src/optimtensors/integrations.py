import os
import tempfile
import copy
import warnings
import torch

try:
    from transformers import TrainerCallback
    from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    # Fallback placeholder so package can be imported without transformers installed
    class TrainerCallback:
        pass
    OPTIMIZER_NAME = "optimizer.pt"
    SCHEDULER_NAME = "scheduler.pt"
    INTEGRATIONS_AVAILABLE = False

try:
    from transformers.utils import check_torch_load_is_safe
except ImportError:
    def check_torch_load_is_safe():
        pass

try:
    from transformers.trainer import reissue_pt_warnings
except ImportError:
    def reissue_pt_warnings(caught_warnings):
        pass

from optimtensors.serde import safe_save_optimizer, safe_load_optimizer


def state_dict_to_cpu(state_dict: dict) -> dict:
    """Moves optimizer state dict tensors to CPU dynamically and ensures contiguous memory."""
    cpu_state_dict = {}
    for k, v in state_dict.items():
        if k == "state":
            cpu_state_dict["state"] = {}
            for param_id, param_state in v.items():
                cpu_param_state = {}
                for sk, sv in param_state.items():
                    if isinstance(sv, torch.Tensor):
                        cpu_param_state[sk] = sv.detach().cpu().clone()
                    else:
                        cpu_param_state[sk] = sv
                cpu_state_dict["state"][param_id] = cpu_param_state
        elif k == "param_groups":
            cpu_state_dict["param_groups"] = copy.deepcopy(v)
        else:
            cpu_state_dict[k] = v
    return cpu_state_dict


class OptimTensorsCallback(TrainerCallback):
    """
    TrainerCallback to save optimizer states securely using optimtensors.
    
    Args:
        mode (str): "dual" (saves optimtensors alongside optimizer.pt) or
                    "strict" (deletes optimizer.pt after successful safe serialization).
        filename (str): Output filename in checkpoint directory.
    """
    def __init__(self, mode="dual", filename="optimizer.optimtensors"):
        if mode not in ["dual", "strict"]:
            raise ValueError("mode must be either 'dual' or 'strict'")
        self.mode = mode
        self.filename = filename

    def on_save(self, args, state, control, **kwargs):
        if not INTEGRATIONS_AVAILABLE:
            return control
            
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return control

        # Only save on rank 0
        if not state.is_world_process_zero:
            return control

        # Distributed guards: no-op with warning for sharded setups
        is_deepspeed = getattr(args, "deepspeed", None) is not None
        is_fsdp = False
        if hasattr(args, "fsdp") and args.fsdp:
            is_fsdp = True
        
        # Check FSDP class dynamically
        model = kwargs.get("model")
        if model is not None:
            model_class_name = type(model).__name__
            if "FullyShardedDataParallel" in model_class_name:
                is_fsdp = True

        if is_deepspeed or is_fsdp:
            warnings.warn(
                "OptimTensorsCallback no-oped: DeepSpeed, FSDP, and sharded optimizer configurations "
                "are not supported in v1 callback. Distributed optimizer checkpointing is handled "
                "via PyTorch Distributed Checkpoint (DCP).",
                UserWarning
            )
            return control

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)

        filepath = os.path.join(checkpoint_dir, self.filename)

        # Move states to CPU dynamically to prepare contiguous buffer layout
        cpu_state = state_dict_to_cpu(optimizer.state_dict())

        # Save atomically using a temp file in the same checkpoint directory
        temp_fd, temp_path = tempfile.mkstemp(dir=checkpoint_dir, suffix=".tmp")
        try:
            os.close(temp_fd)
            optimizer_type = type(optimizer).__name__
            safe_save_optimizer(cpu_state, temp_path, optimizer_type=optimizer_type)
            os.replace(temp_path, filepath)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        # If strict mode, delete standard pickled checkpoints
        if self.mode == "strict":
            optimizer_pt_path = os.path.join(checkpoint_dir, OPTIMIZER_NAME)
            if os.path.exists(optimizer_pt_path):
                os.remove(optimizer_pt_path)
            # Check for binary variant
            optimizer_bin_path = os.path.join(checkpoint_dir, "optimizer.bin")
            if os.path.exists(optimizer_bin_path):
                os.remove(optimizer_bin_path)

        return control


class OptimTensorsTrainerMixin:
    """
    Mixin for overrides inside Trainer to load optimizer states safely using optimtensors.
    """
    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        if checkpoint is None:
            return

        # Delegate distributed configs directly to standard trainer loader
        is_deepspeed = getattr(self, "is_deepspeed_enabled", False)
        is_fsdp = getattr(self, "is_fsdp_enabled", False)
        if is_deepspeed or is_fsdp:
            super()._load_optimizer_and_scheduler(checkpoint)
            return

        filename = "optimizer.optimtensors"
        optimtensors_path = os.path.join(checkpoint, filename)

        if os.path.isfile(optimtensors_path):
            # Load optimizer state dict from secure binary mapping
            state_dict = safe_load_optimizer(optimtensors_path)

            # Move tensors to the correct training device
            device = self.args.device
            for param_state in state_dict.get("state", {}).values():
                for sk, sv in param_state.items():
                    if isinstance(sv, torch.Tensor):
                        param_state[sk] = sv.to(device)

            self.optimizer.load_state_dict(state_dict)

            # Explicitly load learning rate scheduler since optimizer.pt is gone/skipped
            scheduler_path = os.path.join(checkpoint, SCHEDULER_NAME)
            if os.path.isfile(scheduler_path):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    self.lr_scheduler.load_state_dict(
                        torch.load(scheduler_path, weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
            return

        # Fallback to default loading if safe checkpoint is missing
        super()._load_optimizer_and_scheduler(checkpoint)


def load_trainer_optimizer(checkpoint_dir: str, optimizer: torch.optim.Optimizer, device: torch.device = None) -> None:
    """
    Utility helper to manually load optimizer state dicts from safe checkpoint files.
    """
    filename = "optimizer.optimtensors"
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OptimTensors checkpoint not found at: {path}")

    state_dict = safe_load_optimizer(path)
    if device is not None:
        for param_state in state_dict.get("state", {}).values():
            for sk, sv in param_state.items():
                if isinstance(sv, torch.Tensor):
                    param_state[sk] = sv.to(device)
    optimizer.load_state_dict(state_dict)
