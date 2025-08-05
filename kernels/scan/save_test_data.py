import torch
import pickle
from selective_scan_pytorch import selective_scan_ref

def save_test_cases():
    """Save test cases for comparing PyTorch vs CUDA implementations."""
    
    test_cases = []
    
    # Different configurations to test
    configs = [
        (1, 256, 128, 8, "tiny"),
        (2, 512, 256, 16, "small"),
        (4, 768, 512, 16, "medium"),
        (1, 1024, 1024, 32, "large"),
    ]
    
    torch.manual_seed(42)  # For reproducibility
    
    for batch, dim, seq_len, dstate, name in configs:
        # Generate inputs
        u = torch.randn(batch, dim, seq_len, dtype=torch.float32)
        delta = torch.randn(batch, dim, seq_len, dtype=torch.float32).abs()
        A = -torch.rand(dim, dstate, dtype=torch.float32)
        B = torch.randn(batch, dstate, seq_len, dtype=torch.float32)
        C = torch.randn(batch, dstate, seq_len, dtype=torch.float32)
        D = torch.randn(dim, dtype=torch.float32)
        
        # Compute reference output
        y_ref = selective_scan_ref(u, delta, A, B, C, D=D)
        
        test_case = {
            'name': name,
            'inputs': {
                'u': u,
                'delta': delta,
                'A': A,
                'B': B,
                'C': C,
                'D': D,
            },
            'output': y_ref,
            'config': {
                'batch': batch,
                'dim': dim,
                'seq_len': seq_len,
                'dstate': dstate,
            }
        }
        
        test_cases.append(test_case)
        print(f"Generated test case: {name} - Shape: ({batch}, {dim}, {seq_len}, {dstate})")
    
    # Save to file
    torch.save(test_cases, 'test_cases.pt')
    print(f"\nSaved {len(test_cases)} test cases to test_cases.pt")

if __name__ == "__main__":
    save_test_cases()