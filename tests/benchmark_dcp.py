import os
import time
import json
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed.checkpoint as dcp

from optimtensors import SecureFileSystemWriter, SecureFileSystemReader
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader

def run_dcp_benchmark(rank, world_size, temp_dir_native, temp_dir_secure, sync_file, results_queue):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{sync_file}",
        world_size=world_size,
        rank=rank
    )
    
    # Create large dummy state dict for realistic testing (e.g. 50MB optimizer state)
    state_dict = {
        f"param_{rank}": torch.randn(2000, 2000, dtype=torch.float32)
    }
    
    # Warmup Save to eliminate initialization/disk-allocation overhead
    warmup_writer = FileSystemWriter(temp_dir_native)
    dcp.save(state_dict, storage_writer=warmup_writer)
    dist.barrier()
    
    # Benchmark Native DCP Save
    dist.barrier()
    t_start = time.perf_counter()
    native_writer = FileSystemWriter(temp_dir_native)
    dcp.save(state_dict, storage_writer=native_writer)
    dist.barrier()
    native_save_time = time.perf_counter() - t_start
    
    # Benchmark Secure DCP Save
    t_start = time.perf_counter()
    secure_writer = SecureFileSystemWriter(temp_dir_secure)
    dcp.save(state_dict, storage_writer=secure_writer)
    dist.barrier()
    secure_save_time = time.perf_counter() - t_start
    
    # Setup load targets
    load_state_dict = {
        f"param_{rank}": torch.empty(2000, 2000, dtype=torch.float32)
    }
    
    # Benchmark Native DCP Load
    dist.barrier()
    t_start = time.perf_counter()
    native_reader = FileSystemReader(temp_dir_native)
    dcp.load(load_state_dict, storage_reader=native_reader)
    dist.barrier()
    native_load_time = time.perf_counter() - t_start
    
    # Reset load targets
    load_state_dict = {
        f"param_{rank}": torch.empty(2000, 2000, dtype=torch.float32)
    }
    
    # Benchmark Secure DCP Load
    t_start = time.perf_counter()
    secure_reader = SecureFileSystemReader(temp_dir_secure)
    dcp.load(load_state_dict, storage_reader=secure_reader)
    dist.barrier()
    secure_load_time = time.perf_counter() - t_start
    
    # Report back only from rank 0
    if rank == 0:
        results_queue.put({
            "native_save_time": native_save_time,
            "secure_save_time": secure_save_time,
            "native_load_time": native_load_time,
            "secure_load_time": secure_load_time,
        })
        
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    
    with tempfile.TemporaryDirectory() as tmp_native, \
         tempfile.TemporaryDirectory() as tmp_secure:
         
        sync_fd, sync_file = tempfile.mkstemp()
        os.close(sync_fd)
        
        ctx = mp.get_context("spawn")
        results_queue = ctx.Queue()
        
        try:
            mp.spawn(
                run_dcp_benchmark,
                args=(world_size, tmp_native, tmp_secure, sync_file, results_queue),
                nprocs=world_size,
                join=True
            )
            
            # Read results
            res = results_queue.get()
            
            # Read metadata sizes
            native_meta_path = os.path.join(tmp_native, ".metadata")
            secure_meta_path = os.path.join(tmp_secure, "metadata.json")
            
            native_meta_size = os.path.getsize(native_meta_path) if os.path.exists(native_meta_path) else 0
            secure_meta_size = os.path.getsize(secure_meta_path) if os.path.exists(secure_meta_path) else 0
            
            print("\n" + "="*50)
            print("🚀 PYTORCH DISTRIBUTED CHECKPOINT (DCP) BENCHMARK")
            print("="*50)
            print(f"Dataset Size: 2 ranks x 16MB tensor = 32MB Total")
            print(f"Native DCP Save Time (Pickle Metadata): {res['native_save_time']:.5f} s")
            print(f"Secure DCP Save Time (JSON Metadata):   {res['secure_save_time']:.5f} s")
            print(f"DCP Save Overhead Ratio:               {res['secure_save_time']/res['native_save_time']:.2f}x")
            print("-"*50)
            print(f"Native DCP Load Time (Pickle Metadata): {res['native_load_time']:.5f} s")
            print(f"Secure DCP Load Time (JSON Metadata):   {res['secure_load_time']:.5f} s")
            print(f"DCP Load Overhead Ratio:               {res['secure_load_time']/res['native_load_time']:.2f}x")
            print("-"*50)
            print(f"Native Pickle Metadata Size:           {native_meta_size / 1024:.2f} KB")
            print(f"Secure JSON Metadata Size:             {secure_meta_size / 1024:.2f} KB")
            print("="*50 + "\n")
            
        finally:
            if os.path.exists(sync_file):
                os.remove(sync_file)
