import torch
import time
from selective_scan_pytorch import selective_scan_ref

def compare_implementations(cuda_impl=None):
    """Compare PyTorch reference with CUDA implementation."""
    
    test_configs = [
        # (batch, dim, seq_len, dstate, name)
        (1, 512, 256, 16, "Small"),
        (2, 768, 512, 16, "Medium"),
        (4, 1024, 1024, 32, "Large"),
    ]
    
    print("Implementation Comparison")
    print("=" * 80)
    print(f"{'Config':<15} {'PyTorch (ms)':<15} {'CUDA (ms)':<15} {'Speedup':<10} {'Max Error':<10}")
    print("-" * 80)
    
    for batch, dim, seq_len, dstate, name in test_configs:
        # Create test data
        u = torch.randn(batch, dim, seq_len, device='cuda', dtype=torch.float32)
        delta = torch.randn(batch, dim, seq_len, device='cuda').abs()
        A = -torch.rand(dim, dstate, device='cuda')
        B = torch.randn(batch, dstate, seq_len, device='cuda')
        C = torch.randn(batch, dstate, seq_len, device='cuda')
        
        # Time PyTorch
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            y_ref = selective_scan_ref(u, delta, A, B, C)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 10 * 1000
        
        if cuda_impl is not None:
            # Time CUDA
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                y_cuda = cuda_impl(u, delta, A, B, C)
            torch.cuda.synchronize()
            cuda_time = (time.time() - start) / 10 * 1000
            
            # Compare accuracy
            max_error = (y_ref - y_cuda).abs().max().item()
            speedup = pytorch_time / cuda_time
            
            print(f"{name:<15} {pytorch_time:<15.2f} {cuda_time:<15.2f} {speedup:<10.2f} {max_error:<10.2e}")
        else:
            print(f"{name:<15} {pytorch_time:<15.2f} {'N/A':<15} {'N/A':<10} {'N/A':<10}")

if __name__ == "__main__":
    # For now, just test PyTorch
    compare_implementations()