import torch
from selective_scan_pytorch import selective_scan_ref

def profile_memory_usage():
    """Profile memory usage of the sequential implementation."""
    print("Memory Profiling")
    print("=" * 50)
    
    configs = [
        (1, 768, 512, 16),
        (4, 256, 1024, 8),
        (8, 128, 2048, 4),
    ]
    
    for batch, dim, seq_len, dstate in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create inputs
        u = torch.randn(batch, dim, seq_len, device='cuda')
        delta = torch.randn(batch, dim, seq_len, device='cuda')
        A = -torch.rand(dim, dstate, device='cuda')
        B = torch.randn(batch, dstate, seq_len, device='cuda')
        C = torch.randn(batch, dstate, seq_len, device='cuda')
        
        # Measure memory before
        mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Run forward pass
        y = selective_scan_ref(u, delta, A, B, C)
        
        # Measure memory after
        mem_after = torch.cuda.memory_allocated() / 1024**2  # MB
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"Shape: ({batch}, {dim}, {seq_len}, {dstate})")
        print(f"  Memory before: {mem_before:.2f} MB")
        print(f"  Memory after: {mem_after:.2f} MB")
        print(f"  Peak memory: {peak_mem:.2f} MB")
        print(f"  Memory increase: {mem_after - mem_before:.2f} MB")
        print()

if __name__ == "__main__":
    profile_memory_usage()