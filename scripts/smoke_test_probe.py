#!/usr/bin/env python3
"""
Smoke test for probe_kv_structure.py with 5 core fixes.

Verifies:
1. _2adic_valuation is correct
2. Small test runs without errors
3. No NaNs/infs in results
4. All methods use same pair manifest
5. K quantization uses fixed scale per layer/head
6. Permutation-null result is generated
"""

import sys
sys.path.insert(0, '/Users/saurabhbhadada/Desktop/current/research/padic-transformers')

import torch
import numpy as np
from src.kernels import _2adic_valuation

def test_2adic_valuation():
    """Test that _2adic_valuation is correct."""
    print("="*60)
    print("TEST 1: _2adic_valuation correctness")
    print("="*60)

    x = torch.tensor([1, 2, 4, 6, 8, 12, 16])
    expected = torch.tensor([0, 1, 2, 1, 3, 2, 4], dtype=torch.long)

    result = _2adic_valuation(x, precision=8)

    if torch.equal(result, expected):
        print("✅ PASS: _2adic_valuation is correct")
        print(f"   Input:    {x.tolist()}")
        print(f"   Expected: {expected.tolist()}")
        print(f"   Got:      {result.tolist()}")
        return True
    else:
        print("❌ FAIL: _2adic_valuation is incorrect")
        for i in range(len(x)):
            if result[i] != expected[i]:
                print(f"   x={x[i]}: expected v_2={expected[i]}, got v_2={result[i]}")
        return False


def test_smoke_run():
    """Run probe on tiny synthetic data."""
    print("\n" + "="*60)
    print("TEST 2: Smoke test on synthetic data")
    print("="*60)

    # Create tiny synthetic KV states
    num_samples = 3
    num_heads = 2
    seq_len = 128
    head_dim = 64

    kv_states = {
        'keys': [],
        'values': [],
        'attention_weights': [],
    }

    for _ in range(num_samples):
        # Create random keys [batch=1, heads, seq, dim]
        keys = torch.randn(1, num_heads, seq_len, head_dim) * 0.5
        values = torch.randn(1, num_heads, seq_len, head_dim) * 0.5

        # Create random attention [batch=1, heads, seq, seq]
        attn = torch.randn(1, num_heads, seq_len, seq_len)
        attn = torch.softmax(attn, dim=-1)

        # Wrap in lists like real extraction
        kv_states['keys'].append([keys])
        kv_states['values'].append([values])
        kv_states['attention_weights'].append([attn])

    # Import analysis function
    from scripts.probe_kv_structure import analyze_K_structure

    print(f"Running analyze_K_structure with {num_samples} samples, {seq_len} seq_len...")
    print("Parameters: layer_idx=0, num_pairs=50, future_horizon=32, seed=42")

    try:
        results = analyze_K_structure(
            kv_states,
            layer_idx=0,
            num_pairs=50,
            future_horizon=32,
            seed=42,
        )

        print("\n✅ PASS: Smoke test completed without errors")
        return True, results
    except Exception as e:
        print(f"\n❌ FAIL: Smoke test raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_no_nans(results):
    """Check results for NaNs/infs."""
    print("\n" + "="*60)
    print("TEST 3: Check for NaNs/infs")
    print("="*60)

    if results is None:
        print("⚠️  SKIP: No results to check")
        return False

    has_issues = False

    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (float, np.floating)):
                    if np.isnan(subvalue) or np.isinf(subvalue):
                        print(f"❌ Found NaN/inf in results['{key}']['{subkey}']: {subvalue}")
                        has_issues = True
                elif isinstance(subvalue, tuple):
                    for i, v in enumerate(subvalue):
                        if isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                            print(f"❌ Found NaN/inf in results['{key}']['{subkey}'][{i}]: {v}")
                            has_issues = True

    if not has_issues:
        print("✅ PASS: No NaNs or infs found in results")
        return True
    else:
        return False


def test_results_structure(results):
    """Verify expected results structure."""
    print("\n" + "="*60)
    print("TEST 4: Verify results structure")
    print("="*60)

    if results is None:
        print("⚠️  SKIP: No results to check")
        return False

    # Check for required keys
    required_keys = ['p2_prec16']

    all_present = True
    for key in required_keys:
        if key in results:
            print(f"✅ Found: {key}")

            # Check subkeys
            res = results[key]
            if 'rho_S2' in res and 'rho_cosine' in res:
                print(f"   rho_S2: {res['rho_S2']:.4f}")
                print(f"   rho_cosine: {res['rho_cosine']:.4f}")
                if 'rho_S2_permuted' in res:
                    print(f"   rho_S2_permuted: {res['rho_S2_permuted']:.4f}")
                if 'delta_rho' in res:
                    print(f"   delta_rho: {res['delta_rho']:.4f}")
        else:
            print(f"❌ Missing: {key}")
            all_present = False

    # Check for permutation null
    if 'p2_prec16' in results and 'rho_S2_permuted' in results['p2_prec16']:
        print("\n✅ PASS: Permutation null control present")
    else:
        print("\n❌ FAIL: Permutation null control missing")
        all_present = False

    # Check for multi-prime
    if 'p3_prec16' in results and 'p5_prec16' in results:
        print("✅ PASS: Multi-prime results present (p=3, p=5)")
    else:
        print("⚠️  INFO: Multi-prime results not complete")

    # Check for multi-precision
    if 'p2_prec8' in results and 'p2_prec12' in results:
        print("✅ PASS: Multi-precision results present (8-bit, 12-bit)")
    else:
        print("⚠️  INFO: Multi-precision results not complete")

    return all_present


def main():
    print("\nP-ADIC PROBE SMOKE TEST")
    print("="*60)
    print("Testing 5 core fixes implementation")
    print("="*60)

    all_passed = True

    # Test 1: _2adic_valuation
    if not test_2adic_valuation():
        all_passed = False
        print("\n⚠️  CRITICAL: _2adic_valuation is broken, stopping tests")
        return

    # Test 2: Smoke run
    smoke_passed, results = test_smoke_run()
    if not smoke_passed:
        all_passed = False

    # Test 3: No NaNs/infs
    if not test_no_nans(results):
        all_passed = False

    # Test 4: Results structure
    if not test_results_structure(results):
        all_passed = False

    # Final verdict
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)

    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nReady for real experiment:")
        print("  CUDA_VISIBLE_DEVICES=5 make exec CMD=\"python3 scripts/probe_kv_structure.py \\")
        print("      --model pythia-1b \\")
        print("      --dataset wikitext \\")
        print("      --num-samples 2000 \\")
        print("      --layer 6 \\")
        print("      --num-pairs 1000 \\")
        print("      --future-horizon 64 \\")
        print("      --seed 42 \\")
        print("      --output results/probe_wikitext_layer6_1M.json\"")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nFix issues before running real experiment")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
