"""
Core 2-adic arithmetic operations for neural networks.

2-adic numbers are represented modulo 2^precision, providing:
- Natural binary alignment with hardware
- Hierarchical/ultrametric structure
- Memory-efficient representation
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class PadicConfig:
    """Configuration for 2-adic operations."""
    def __init__(self, prime: int = 2, precision: int = 8):
        assert prime == 2, "Currently only 2-adic (p=2) is supported"
        assert 1 <= precision <= 32, "Precision must be between 1 and 32 bits"
        self.prime = prime
        self.precision = precision
        self.modulus = 2 ** precision


def float_to_2adic(x: torch.Tensor, precision: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert floating point tensor to 2-adic representation with dynamic range.

    Uses symmetric quantization: finds max absolute value and scales accordingly.
    This preserves the actual range of values instead of clamping to [-1, 1].

    Process:
    1. Find max absolute value for scaling
    2. Scale to integer range based on actual data range
    3. Take modulo 2^precision
    4. Return quantized tensor and scale factor

    Args:
        x: Input tensor (float32 or bfloat16)
        precision: Number of bits for 2-adic representation (1-32)

    Returns:
        Tuple of (quantized tensor as int32, scale factor as float tensor)

    Example:
        >>> x = torch.tensor([0.5, -0.25, 1.0])
        >>> x_2adic, scale = float_to_2adic(x, precision=8)
        >>> x_2adic
        tensor([128, 192, 0], dtype=torch.int32)
    """
    modulus = 2 ** precision

    # Dynamic range quantization: use actual max absolute value
    # This preserves the range of K/V values in transformers
    max_val = torch.abs(x).max()

    # Choose appropriate dtype based on precision
    if precision <= 8:
        dtype = torch.uint8
    elif precision <= 16:
        dtype = torch.int16
    else:
        dtype = torch.int32

    # Avoid division by zero - use small epsilon for numerical stability
    eps = 1e-8
    if max_val.item() < eps:
        return torch.zeros_like(x, dtype=dtype), torch.tensor(1.0, device=x.device, dtype=torch.float32)

    # Add small epsilon to prevent exact division by zero
    max_val = max_val + eps

    # Scale to [-1, 1] based on actual range, then to [0, modulus-1]
    x_normalized = x / max_val  # Now in [-1, 1]
    x_scaled = ((x_normalized + 1.0) * (modulus / 2.0))

    # Clamp to [0, modulus-1] to prevent wraparound bug
    # Without this, max values (1.0) map to modulus, which wraps to 0 (maps to -1.0)!
    x_scaled = torch.clamp(x_scaled, 0, modulus - 1)

    # Convert to appropriate integer type
    x_int = x_scaled.to(torch.int32)  # Use int32 for intermediate computation
    x_2adic = (x_int % modulus).to(dtype)  # Cast to target dtype

    return x_2adic, max_val


def _2adic_to_float(x: torch.Tensor, precision: int = 8, scale: torch.Tensor = None) -> torch.Tensor:
    """
    Convert 2-adic representation back to floating point.

    Inverse operation of float_to_2adic. Requires the scale factor from quantization.

    Args:
        x: Integer tensor with 2-adic representation
        precision: Number of bits used in 2-adic representation
        scale: Scale factor (max absolute value) from quantization

    Returns:
        Float tensor in original range

    Example:
        >>> x_2adic = torch.tensor([128, 192, 0], dtype=torch.int32)
        >>> scale = torch.tensor(1.0)
        >>> x_float = _2adic_to_float(x_2adic, precision=8, scale=scale)
        >>> x_float
        tensor([0.0000, 0.5000, -1.0000])
    """
    modulus = 2 ** precision

    # Convert to float first to avoid dtype issues with modulo
    # (e.g., uint8 % 256 would wrap modulus to 0 causing ZeroDivisionError)
    x_float = x.to(torch.float32)

    # Ensure input is in valid range [0, modulus-1]
    x_float = x_float % modulus

    # Scale back to [-1, 1]
    x_normalized = (x_float / (modulus / 2.0)) - 1.0

    # Scale back to original range
    if scale is not None:
        x_normalized = x_normalized * scale

    return x_normalized


def float_to_2adic_signed(x: torch.Tensor, precision: int = 8) -> torch.Tensor:
    """
    Convert float to 2-adic with signed representation (preserves sign bit).

    Alternative representation that keeps negative numbers negative.

    Args:
        x: Input tensor (float)
        precision: Number of bits for 2-adic representation

    Returns:
        Signed integer tensor
    """
    modulus = 2 ** precision
    half_modulus = 2 ** (precision - 1)

    # Normalize and scale
    x_normalized = torch.clamp(x, -1.0, 1.0)
    x_scaled = x_normalized * half_modulus

    # Round to integer
    x_int = torch.round(x_scaled).to(torch.int32)

    return x_int


def _2adic_to_float_signed(x: torch.Tensor, precision: int = 8) -> torch.Tensor:
    """Convert signed 2-adic back to float."""
    half_modulus = 2 ** (precision - 1)

    # Handle wraparound for signed integers
    x_float = x.to(torch.float32)
    x_normalized = x_float / half_modulus

    return torch.clamp(x_normalized, -1.0, 1.0)


def _padic_valuation(x: torch.Tensor, prime: int = 2, precision: int = 8) -> torch.Tensor:
    """
    Compute p-adic valuation: highest power of prime p dividing x.

    For each element independently: v_p(x) = max{n : p^n | x}

    Args:
        x: Integer tensor
        prime: Prime number (2, 3, 5, etc.)
        precision: Maximum valuation to compute

    Returns:
        Tensor of valuations (one per element)

    Example:
        >>> x = torch.tensor([1, 2, 4, 6, 8, 12, 16])
        >>> _padic_valuation(x, prime=2, precision=8)
        tensor([0, 1, 2, 1, 3, 2, 4])
        >>> _padic_valuation(x, prime=3, precision=8)
        tensor([0, 0, 0, 1, 0, 1, 0])
    """
    x_abs = torch.abs(x)

    # Handle zeros first (infinite valuation, capped at precision)
    zero_mask = (x_abs == 0)

    # Initialize valuation for each element
    valuation = torch.zeros_like(x_abs, dtype=torch.long)

    # Create working copy
    temp = x_abs.clone()

    # For each power of prime, check divisibility
    for i in range(precision):
        # Check which elements are CURRENTLY divisible by prime and non-zero
        is_divisible = (temp % prime == 0) & (~zero_mask)

        # If no more divisible elements, we're done
        if not is_divisible.any():
            break

        # Increment valuation ONLY for currently divisible elements
        valuation[is_divisible] += 1

        # Divide ONLY the divisible elements by prime
        temp[is_divisible] = temp[is_divisible] // prime

    # Set zero valuations to precision
    valuation[zero_mask] = precision

    return valuation


def _2adic_valuation(x: torch.Tensor, precision: int = 8) -> torch.Tensor:
    """
    Compute 2-adic valuation: highest power of 2 dividing x.

    Wrapper around _padic_valuation for backwards compatibility.
    """
    return _padic_valuation(x, prime=2, precision=precision)


def ultrametric_distance(x: torch.Tensor, y: torch.Tensor, precision: int = 8) -> torch.Tensor:
    """
    Compute 2-adic ultrametric distance between tensors.

    d_2(x, y) = 2^(-v_2(x - y))

    Two numbers are "close" in 2-adic metric if their difference
    is divisible by a high power of 2.

    Args:
        x, y: Integer tensors (2-adic representation)
        precision: Bit precision

    Returns:
        Distance tensor (float), where smaller = more similar

    Example:
        >>> x = torch.tensor([8, 16])
        >>> y = torch.tensor([10, 18])
        >>> ultrametric_distance(x, y)
        tensor([0.5000, 0.5000])  # Both differ by 2 = 2^1
    """
    diff = x - y
    valuation = _2adic_valuation(diff, precision)

    # Distance = 2^(-valuation)
    # Higher valuation = smaller distance = more similar
    distance = 2.0 ** (-valuation.to(torch.float32))

    return distance


def _2adic_norm(x: torch.Tensor, precision: int = 8) -> torch.Tensor:
    """
    Compute 2-adic norm: |x|_2 = 2^(-v_2(x))

    Args:
        x: Integer tensor (2-adic representation)
        precision: Bit precision

    Returns:
        Norm tensor (float)
    """
    valuation = _2adic_valuation(x, precision)
    norm = 2.0 ** (-valuation.to(torch.float32))
    return norm


# Utility functions for debugging and visualization

def _2adic_to_binary_string(x: int, precision: int = 8) -> str:
    """Convert 2-adic integer to binary string for visualization."""
    return format(x % (2**precision), f'0{precision}b')


def visualize_2adic(x: torch.Tensor, precision: int = 8, max_elements: int = 10):
    """
    Print human-readable visualization of 2-adic tensor.

    Args:
        x: 2-adic integer tensor
        precision: Bit precision
        max_elements: Maximum number of elements to display
    """
    x_flat = x.flatten()[:max_elements]

    print(f"2-adic tensor (precision={precision} bits):")
    print(f"Shape: {x.shape}")
    print(f"\nFirst {min(len(x_flat), max_elements)} elements:")
    print(f"{'Decimal':<10} {'Binary':<{precision+2}} {'Float equiv':<12} {'Valuation'}")
    print("-" * (precision + 36))

    for val in x_flat:
        val_int = val.item()
        binary = _2adic_to_binary_string(val_int, precision)
        float_equiv = _2adic_to_float(torch.tensor([val_int]), precision).item()
        valuation = _2adic_valuation(torch.tensor([val_int]), precision).item()
        print(f"{val_int:<10} {binary:<{precision+2}} {float_equiv:<12.6f} {valuation}")


# Gradient-friendly versions for training

class Float2Adic(torch.autograd.Function):
    """
    Differentiable 2-adic conversion with straight-through estimator.

    Forward: quantize to 2-adic
    Backward: pass gradient through (STE)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, precision: int = 8):
        ctx.precision = precision
        return float_to_2adic(x, precision)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradient unchanged
        return grad_output, None


def float_to_2adic_differentiable(x: torch.Tensor, precision: int = 8) -> torch.Tensor:
    """Differentiable version of float_to_2adic for use in training."""
    return Float2Adic.apply(x, precision)
