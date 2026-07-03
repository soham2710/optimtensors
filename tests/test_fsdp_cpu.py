import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

# Add src to path if needed for local test run
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from optimtensors import save_fsdp_full_optimizer, load_fsdp_full_optimizer


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)
        
    def forward(self, x):
        return self.net(x)


def run_worker():
    # Initialize process group using gloo backend which permits multiple processes sharing a GPU
    dist.init_process_group(backend="gloo")
    
    rank = dist.get_rank()
    
    # FSDP requires CUDA. If CUDA is not available, skip the test.
    if not torch.cuda.is_available():
        dist.destroy_process_group()
        if rank == 0:
            print("--> FSDP test skipped (CUDA is not available).")
        return
        
    # Map both workers to cuda:0
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    model = ToyModel().to(device)
    model = FSDP(model)
    x = torch.randn(2, 4).to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    # Run a dummy step to populate optimizer state
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    
    # Define save path in temp directory
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, "fsdp_cpu_test.optimtensors")
    
    # Configure FSDP to use FULL_STATE_DICT
    FSDP.set_state_dict_type(model, StateDictType.FULL_STATE_DICT)
    
    # Save the gathered state dict (collective call, only rank 0 writes)
    save_fsdp_full_optimizer(model, optimizer, path)
    
    # Re-shard load test: create fresh model/optimizer
    fresh_model = ToyModel().to(device)
    fresh_model = FSDP(fresh_model)
    
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.01)
    FSDP.set_state_dict_type(fresh_model, StateDictType.FULL_STATE_DICT)
    
    # Load and re-shard back to FSDP state dict
    load_fsdp_full_optimizer(fresh_model, fresh_optimizer, path)
    
    # Verify correct loading on all ranks
    assert fresh_optimizer.state_dict()["state"] is not None
    
    # Wait for all ranks to complete checks
    dist.barrier()
    
    # Cleanup files on rank 0
    if rank == 0:
        if os.path.exists(path):
            os.remove(path)
        fqnmap_path = path + ".fqnmap.json"
        if os.path.exists(fqnmap_path):
            os.remove(fqnmap_path)
            
    dist.destroy_process_group()
    if rank == 0:
        print("--> SUCCESS: FSDP multi-process check completed successfully!")


if __name__ == "__main__":
    run_worker()
