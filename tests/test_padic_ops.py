"""Unit tests for 2-adic operations."""

import pytest
import torch
import sys
sys.path.insert(0, '/workspace/kv-probe/src')

from geometries.padic import (
    float_to_2adic,
    _2adic_to_float,
    float_to_2adic_signed,
    _2adic_to_float_signed,
    ultrametric_distance,
    _2adic_valuation,
    _2adic_norm,
)


class TestBasicConversion:
    """Test basic float <-> 2-adic conversion."""

    def test_simple_conversion_roundtrip(self):
        """Test that float -> 2adic -> float preserves values approximately."""
        x = torch.tensor([0.5, -0.5, 0.0, 1.0, -1.0])

        # Convert to 2-adic and back
        x_2adic = float_to_2adic(x, precision=8)
        x_reconstructed = _2adic_to_float(x_2adic, precision=8)

        # Should be close (within quantization error)
        torch.testing.assert_close(x, x_reconstructed, atol=0.01, rtol=0.01)

    def test_precision_levels(self):
        """Test different precision levels."""
        x = torch.tensor([0.123, 0.456, 0.789])

        for precision in [4, 8, 12, 16]:
            x_2adic = float_to_2adic(x, precision=precision)
            x_reconstructed = _2adic_to_float(x_2adic, precision=precision)

            # Higher precision should give better accuracy
            max_error = (x - x_reconstructed).abs().max()
            expected_error = 2.0 / (2 ** precision)  # Rough estimate

            assert max_error < expected_error * 2, \
                f"Precision {precision} has error {max_error}, expected < {expected_error*2}"

    def test_signed_conversion(self):
        """Test signed 2-adic conversion."""
        x = torch.tensor([-0.8, -0.4, 0.0, 0.4, 0.8])

        x_2adic = float_to_2adic_signed(x, precision=8)
        x_reconstructed = _2adic_to_float_signed(x_2adic, precision=8)

        torch.testing.assert_close(x, x_reconstructed, atol=0.02, rtol=0.02)

        # Check that negative values have negative representation
        assert (x_2adic[x < 0] < 0).all(), "Negative values should have negative 2-adic representation"

    def test_output_dtype(self):
        """Test that output has correct dtype."""
        x = torch.tensor([0.1, 0.2, 0.3])

        x_2adic = float_to_2adic(x, precision=8)
        assert x_2adic.dtype == torch.int32, f"Expected int32, got {x_2adic.dtype}"

    def test_batch_processing(self):
        """Test that batched tensors work correctly."""
        x = torch.randn(4, 8, 128)  # Batch of matrices

        x_2adic = float_to_2adic(x, precision=8)
        x_reconstructed = _2adic_to_float(x_2adic, precision=8)

        assert x_2adic.shape == x.shape, "Shape should be preserved"
        torch.testing.assert_close(x.clamp(-1, 1), x_reconstructed, atol=0.02, rtol=0.02)


class Test2adicValuation:
    """Test 2-adic valuation function."""

    def test_known_valuations(self):
        """Test valuation on known values."""
        # 8 = 2^3, valuation should be 3
        x = torch.tensor([8])
        val = _2adic_valuation(x, precision=8)
        assert val.item() == 3, f"v_2(8) should be 3, got {val.item()}"

        # 12 = 4 * 3 = 2^2 * 3, valuation should be 2
        x = torch.tensor([12])
        val = _2adic_valuation(x, precision=8)
        assert val.item() == 2, f"v_2(12) should be 2, got {val.item()}"

        # 15 = 15 (odd), valuation should be 0
        x = torch.tensor([15])
        val = _2adic_valuation(x, precision=8)
        assert val.item() == 0, f"v_2(15) should be 0, got {val.item()}"

    def test_zero_valuation(self):
        """Test that valuation of 0 is infinity (represented as precision)."""
        x = torch.tensor([0])
        val = _2adic_valuation(x, precision=8)
        assert val.item() == 8, f"v_2(0) should be {8}, got {val.item()}"

    def test_powers_of_two(self):
        """Test valuation for powers of 2."""
        for i in range(1, 8):
            x = torch.tensor([2 ** i])
            val = _2adic_valuation(x, precision=16)
            assert val.item() == i, f"v_2(2^{i}) should be {i}, got {val.item()}"


class TestUltrametricDistance:
    """Test ultrametric distance function."""

    def test_distance_symmetry(self):
        """Test that d(x,y) = d(y,x)."""
        x = torch.tensor([10, 20, 30])
        y = torch.tensor([15, 22, 35])

        d_xy = ultrametric_distance(x, y, precision=8)
        d_yx = ultrametric_distance(y, x, precision=8)

        torch.testing.assert_close(d_xy, d_yx)

    def test_zero_distance(self):
        """Test that d(x,x) = 0 (or very small)."""
        x = torch.tensor([10, 20, 30])

        d = ultrametric_distance(x, x, precision=8)

        # Distance should be very small (close to 0)
        assert d.max() < 1e-6, f"d(x,x) should be ~0, got {d}"

    def test_ultrametric_inequality(self):
        """Test strong triangle inequality: d(x,z) <= max(d(x,y), d(y,z))."""
        x = torch.tensor([8, 16, 24])
        y = torch.tensor([10, 18, 26])
        z = torch.tensor([12, 20, 28])

        d_xz = ultrametric_distance(x, z, precision=8)
        d_xy = ultrametric_distance(x, y, precision=8)
        d_yz = ultrametric_distance(y, z, precision=8)

        max_d = torch.maximum(d_xy, d_yz)

        # Ultrametric property: d(x,z) <= max(d(x,y), d(y,z))
        assert (d_xz <= max_d + 1e-6).all(), "Ultrametric inequality violated"

    def test_hierarchical_clustering(self):
        """Test that ultrametric distance groups similar numbers."""
        # Numbers differing by powers of 2
        x = torch.tensor([16, 18, 32])  # 16 and 18 differ by 2, should be closer

        d_16_18 = ultrametric_distance(x[0:1], x[1:2])
        d_16_32 = ultrametric_distance(x[0:1], x[2:3])

        # 16 and 18 differ by 2 = 2^1, distance = 2^(-1) = 0.5
        # 16 and 32 differ by 16 = 2^4, distance = 2^(-4) = 0.0625

        assert d_16_32 < d_16_18, "Numbers with higher power-of-2 difference should be closer in 2-adic metric"


class TestNorm:
    """Test 2-adic norm."""

    def test_norm_properties(self):
        """Test basic norm properties."""
        x = torch.tensor([8, 12, 15])

        norm = _2adic_norm(x, precision=8)

        # Norm should be positive
        assert (norm > 0).all(), "Norm should be positive"

        # Norm of 0 should be 0 (or very small)
        norm_zero = _2adic_norm(torch.tensor([0]), precision=8)
        assert norm_zero.item() < 1e-6, "Norm of 0 should be ~0"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extreme_values(self):
        """Test with extreme values."""
        x = torch.tensor([-1.0, -0.99, 0.99, 1.0])

        x_2adic = float_to_2adic(x, precision=8)
        x_reconstructed = _2adic_to_float(x_2adic, precision=8)

        torch.testing.assert_close(x, x_reconstructed, atol=0.02, rtol=0.02)

    def test_small_values(self):
        """Test with very small values near zero."""
        x = torch.tensor([-0.01, -0.001, 0.0, 0.001, 0.01])

        x_2adic = float_to_2adic(x, precision=16)  # Higher precision for small values
        x_reconstructed = _2adic_to_float(x_2adic, precision=16)

        torch.testing.assert_close(x, x_reconstructed, atol=0.001, rtol=0.1)

    def test_empty_tensor(self):
        """Test with empty tensor."""
        x = torch.tensor([])

        x_2adic = float_to_2adic(x, precision=8)
        assert x_2adic.shape == torch.Size([0]), "Empty tensor should remain empty"

    def test_single_element(self):
        """Test with single element."""
        x = torch.tensor([0.5])

        x_2adic = float_to_2adic(x, precision=8)
        x_reconstructed = _2adic_to_float(x_2adic, precision=8)

        torch.testing.assert_close(x, x_reconstructed, atol=0.01, rtol=0.01)


class TestGPUCompat:
    """Test GPU compatibility (if available)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_conversion(self):
        """Test that operations work on CUDA tensors."""
        x = torch.tensor([0.1, 0.2, 0.3]).cuda()

        x_2adic = float_to_2adic(x, precision=8)
        x_reconstructed = _2adic_to_float(x_2adic, precision=8)

        assert x_2adic.is_cuda, "Output should be on CUDA"
        torch.testing.assert_close(x, x_reconstructed, atol=0.02, rtol=0.02)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
