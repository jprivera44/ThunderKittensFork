import torch
import torch.nn.functional as F
from selective_scan_pytorch import selective_scan_ref, selective_scan_easy

def test_basic_shapes():
    """Test that output shapes are correct."""
    print("Testing basic shapes...")
    
    batch, dim, seq_len, dstate = 2, 64, 128, 16
    
    # Create inputs
    u = torch.randn(batch, dim, seq_len)
    delta = torch.randn(batch, dim, seq_len)
    A = -torch.rand(dim, dstate)  # Negative for stability
    B = torch.randn(batch, dstate, seq_len)
    C = torch.randn(batch, dstate, seq_len)
    
    # Run
    y = selective_scan_easy(u, delta, A, B, C)
    
    # Check shape
    assert y.shape == (batch, dim, seq_len), f"Expected {(batch, dim, seq_len)}, got {y.shape}"
    print("✓ Shape test passed")


def test_causal_property():
    """Test that the operation is causal (no future lookups)."""
    print("\nTesting causality...")
    
    batch, dim, seq_len, dstate = 1, 4, 10, 2
    
    # Create inputs
    u = torch.randn(batch, dim, seq_len)
    delta = torch.randn(batch, dim, seq_len)
    A = -torch.rand(dim, dstate)
    B = torch.randn(batch, dstate, seq_len)
    C = torch.randn(batch, dstate, seq_len)
    
    # Run full sequence
    y_full = selective_scan_easy(u, delta, A, B, C)
    
    # Run partial sequence
    y_partial = selective_scan_easy(
        u[:, :, :5], 
        delta[:, :, :5],
        A,
        B[:, :, :5],
        C[:, :, :5]
    )
    
    # First 5 outputs should match
    assert torch.allclose(y_full[:, :, :5], y_partial, rtol=1e-5)
    print("✓ Causality test passed")


def test_with_gating():
    """Test with multiplicative gating (z parameter)."""
    print("\nTesting with gating...")
    
    batch, dim, seq_len, dstate = 2, 32, 64, 8
    
    # Create inputs
    u = torch.randn(batch, dim, seq_len)
    delta = torch.randn(batch, dim, seq_len)
    A = -torch.rand(dim, dstate)
    B = torch.randn(batch, dstate, seq_len)
    C = torch.randn(batch, dstate, seq_len)
    z = torch.randn(batch, dim, seq_len)
    
    # Run with gating
    y = selective_scan_ref(u, delta, A, B, C, z=z)
    
    assert y.shape == (batch, dim, seq_len)
    print("✓ Gating test passed")

def test_gradient_flow():
    """Ensure gradients flow properly through the scan."""
    print("\nTesting gradient flow...")
    
    batch, dim, seq_len, dstate = 2, 16, 32, 4
    
    # Create inputs
    u = torch.randn(batch, dim, seq_len, requires_grad=True)
    delta = torch.randn(batch, dim, seq_len, requires_grad=True)
    A = -torch.rand(dim, dstate, requires_grad=True)
    B = torch.randn(batch, dstate, seq_len, requires_grad=True)
    C = torch.randn(batch, dstate, seq_len, requires_grad=True)
    
    # Forward pass
    y = selective_scan_easy(u, delta, A, B, C)
    
    # Check that output requires grad
    assert y.requires_grad, "Output doesn't require grad"
    
    # Do a simple backward to verify it doesn't crash
    loss = y.sum()
    loss.backward()
    
    print("✓ Gradient flow test passed")


def test_compare_with_mamba():
    """Compare with Mamba's implementation if available."""
    print("\nTesting comparison with Mamba...")
    
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        
        # Setup inputs matching Mamba's expected format
        batch, dim, seq_len, dstate = 2, 64, 256, 16
        
        u = torch.randn(batch, dim, seq_len, device='cuda', dtype=torch.float32)
        delta = torch.randn(batch, dim, seq_len, device='cuda', dtype=torch.float32)
        A = -torch.rand(dim, dstate, device='cuda', dtype=torch.float32)
        B = torch.randn(batch, dstate, seq_len, device='cuda', dtype=torch.float32)
        C = torch.randn(batch, dstate, seq_len, device='cuda', dtype=torch.float32)
        D = torch.randn(dim, device='cuda', dtype=torch.float32)
        z = torch.randn(batch, dim, seq_len, device='cuda', dtype=torch.float32)
        delta_bias = torch.randn(dim, device='cuda', dtype=torch.float32)
        
        # Mamba's implementation
        y_mamba = selective_scan_fn(
            u, delta, A, B, C, D, z=z,
            delta_bias=delta_bias, delta_softplus=True
        )
        
        # Our implementation
        y_ours = selective_scan_ref(
            u.cpu(), delta.cpu(), A.cpu(), B.cpu(), C.cpu(),
            D=D.cpu(), z=z.cpu(), delta_bias=delta_bias.cpu(),
            delta_softplus=True
        )
        
        # Compare (allowing for small numerical differences)
        assert torch.allclose(
            y_mamba.cpu(), y_ours, rtol=1e-3, atol=1e-3
        ), "Outputs don't match Mamba's implementation!"
        
        print("✓ Mamba comparison test passed")
        
    except ImportError:
        print("⚠ Mamba not installed, skipping comparison test")
        print("  Install with: pip install mamba-ssm")


def benchmark_implementation():
    """Benchmark the PyTorch implementation."""
    print("\n" + "="*50)
    print("BENCHMARKING")
    print("="*50)
    
    import time
    
    configs = [
        (1, 768, 512, 16, "BERT-base like"),
        (1, 1024, 512, 16, "BERT-large like"),
        (4, 256, 1024, 8, "Batch=4, long sequence"),
        (8, 128, 2048, 4, "Batch=8, very long sequence"),
    ]
    
    for batch, dim, seq_len, dstate, desc in configs:
        # Create inputs on GPU
        u = torch.randn(batch, dim, seq_len, device='cuda')
        delta = torch.randn(batch, dim, seq_len, device='cuda')
        A = -torch.rand(dim, dstate, device='cuda')
        B = torch.randn(batch, dstate, seq_len, device='cuda')
        C = torch.randn(batch, dstate, seq_len, device='cuda')
        
        # Warmup
        for _ in range(10):
            _ = selective_scan_easy(u, delta, A, B, C)
        torch.cuda.synchronize()
        
        # Time it
        start = time.time()
        num_iters = 50
        for _ in range(num_iters):
            y = selective_scan_easy(u, delta, A, B, C)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        ms_per_iter = (elapsed / num_iters) * 1000
        print(f"{desc:25} | Shape: ({batch}, {dim}, {seq_len}, {dstate}) | {ms_per_iter:.2f} ms/iter")


if __name__ == "__main__":
    # Run all tests
    test_basic_shapes()
    test_causal_property()
    test_with_gating()
    test_gradient_flow()
    test_compare_with_mamba()
    
    # Benchmark
    benchmark_implementation()
    
    print("\n✅ All tests passed!")