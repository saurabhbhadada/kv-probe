#!/usr/bin/env python3
"""Regression test for _2adic_valuation fix."""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import torch
from src.kernels.padic_ops import _2adic_valuation

def test_2adic_valuation():
    """Test that _2adic_valuation computes correctly for each element independently."""
    
    # Test case from user
    x = torch.tensor([1, 2, 4, 6, 8, 12, 16])
    expected = torch.tensor([
        0,  # 1 = 2^0 * 1
        1,  # 2 = 2^1 * 1
        2,  # 4 = 2^2 * 1
        1,  # 6 = 2^1 * 3
        3,  # 8 = 2^3 * 1
        2,  # 12 = 2^2 * 3
        4,  # 16 = 2^4 * 1
    ], dtype=torch.long)
    
    result = _2adic_valuation(x, precision=8)
    
    print("Input:    ", x.tolist())
    print("Expected: ", expected.tolist())
    print("Got:      ", result.tolist())
    
    if torch.equal(result, expected):
        print("\n✅ PASS: _2adic_valuation is correct")
        return True
    else:
        print("\n❌ FAIL: _2adic_valuation is incorrect")
        for i in range(len(x)):
            if result[i] != expected[i]:
                print(f"  x={x[i]}: expected v_2={expected[i]}, got v_2={result[i]}")
        return False

    # Additional test: simple case from user's bug report
    print("\n" + "="*60)
    x2 = torch.tensor([2, 8])
    expected2 = torch.tensor([1, 3], dtype=torch.long)
    result2 = _2adic_valuation(x2, precision=8)
    
    print("Input:    ", x2.tolist())
    print("Expected: ", expected2.tolist())
    print("Got:      ", result2.tolist())
    
    if torch.equal(result2, expected2):
        print("✅ PASS: Simple case [2, 8] correct")
        return True
    else:
        print("❌ FAIL: Simple case [2, 8] incorrect")
        return False

if __name__ == "__main__":
    success = test_2adic_valuation()
    sys.exit(0 if success else 1)
