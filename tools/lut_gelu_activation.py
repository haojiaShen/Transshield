#!/usr/bin/env python3
"""
Piecewise Linear GELU Activation for SPU (Hawk/Tabula approach)

This module implements a lookup table (LUT) based GELU activation function
that can be used in SPU secure inference. The approach uses piecewise linear
interpolation to approximate the GELU function, which is much more MPC-friendly
than the exact GELU implementation.

Key advantages:
1. Much closer to exact GELU than fixed_square activation
2. MPC-friendly: only requires comparisons and linear interpolation
3. Configurable precision via number of segments
4. Can be used as a drop-in replacement for fixed_square

Reference:
- Hawk: Accurate and Fast Privacy-Preserving Machine Learning Using Secure Lookup Table Computation (2024)
- Tabula: Efficiently Computing Nonlinear Activation Functions for Secure NN Inference (2022)
"""

import jax
import jax.numpy as jnp


def gelu_exact(x):
    """Exact GELU implementation."""
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def gelu_piecewise_linear(x, num_segments=16, x_min=-8.0, x_max=8.0):
    """
    Piecewise linear approximation of GELU using lookup table approach.
    
    This function approximates GELU using piecewise linear interpolation,
    which is much more MPC-friendly than the exact GELU implementation.
    
    Args:
        x: Input tensor
        num_segments: Number of linear segments (default: 16)
        x_min: Minimum value for lookup table range (default: -8.0)
        x_max: Maximum value for lookup table range (default: 8.0)
    
    Returns:
        Approximated GELU output
    """
    # Create breakpoints and values for the lookup table
    breakpoints = jnp.linspace(x_min, x_max, num_segments + 1)
    values = gelu_exact(breakpoints)
    
    # Use piecewise linear interpolation
    return jnp.interp(x, breakpoints, values)


def create_gelu_lut(num_segments=16, x_min=-8.0, x_max=8.0):
    """
    Create a GELU lookup table for use in SPU inference.
    
    Args:
        num_segments: Number of linear segments
        x_min: Minimum value for lookup table range
        x_max: Maximum value for lookup table range
    
    Returns:
        Tuple of (breakpoints, values) for the lookup table
    """
    breakpoints = jnp.linspace(x_min, x_max, num_segments + 1)
    values = gelu_exact(breakpoints)
    return breakpoints, values


def gelu_lut_from_table(x, breakpoints, values):
    """
    Apply GELU using precomputed lookup table.
    
    Args:
        x: Input tensor
        breakpoints: Precomputed breakpoints from create_gelu_lut
        values: Precomputed values from create_gelu_lut
    
    Returns:
        Approximated GELU output
    """
    return jnp.interp(x, breakpoints, values)


# Precomputed lookup tables for common configurations
GELU_LUT_16 = create_gelu_lut(num_segments=16, x_min=-8.0, x_max=8.0)
GELU_LUT_32 = create_gelu_lut(num_segments=32, x_min=-8.0, x_max=8.0)


def gelu_lut_16(x):
    """GELU with 16 segments (fast, good accuracy)."""
    return gelu_lut_from_table(x, *GELU_LUT_16)


def gelu_lut_32(x):
    """GELU with 32 segments (slower, better accuracy)."""
    return gelu_lut_from_table(x, *GELU_LUT_32)


if __name__ == "__main__":
    # Test the implementations
    import numpy as np
    
    print("=== GELU LUT Activation Module ===")
    
    # Test inputs
    x = jnp.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    
    # Compute outputs
    y_exact = gelu_exact(x)
    y_lut_16 = gelu_lut_16(x)
    y_lut_32 = gelu_lut_32(x)
    
    print(f"\nInput\t\tExact GELU\tLUT-16\t\tLUT-32")
    print("-" * 60)
    for i in range(len(x)):
        print(f"{x[i]:.1f}\t\t{y_exact[i]:.4f}\t\t{y_lut_16[i]:.4f}\t\t{y_lut_32[i]:.4f}")
    
    # Compute errors
    error_16 = jnp.mean(jnp.abs(y_exact - y_lut_16))
    error_32 = jnp.mean(jnp.abs(y_exact - y_lut_32))
    
    print(f"\nMean Absolute Error:")
    print(f"  LUT-16: {error_16:.6f}")
    print(f"  LUT-32: {error_32:.6f}")
    
    # Test with random inputs
    print(f"\n=== Random Input Test ===")
    np.random.seed(42)
    x_random = jnp.array(np.random.randn(1000) * 2)
    
    y_exact_random = gelu_exact(x_random)
    y_lut_16_random = gelu_lut_16(x_random)
    y_lut_32_random = gelu_lut_32(x_random)
    
    error_16_random = jnp.mean(jnp.abs(y_exact_random - y_lut_16_random))
    error_32_random = jnp.mean(jnp.abs(y_exact_random - y_lut_32_random))
    
    print(f"Mean Absolute Error (1000 random inputs):")
    print(f"  LUT-16: {error_16_random:.6f}")
    print(f"  LUT-32: {error_32_random:.6f}")
    
    # Test gradient behavior
    print(f"\n=== Gradient Behavior ===")
    x_grad = jnp.array([0.5, 1.0, 1.5, 2.0])
    
    grad_exact = jax.grad(lambda x: jnp.sum(gelu_exact(x)))(x_grad)
    grad_lut_16 = jax.grad(lambda x: jnp.sum(gelu_lut_16(x)))(x_grad)
    grad_lut_32 = jax.grad(lambda x: jnp.sum(gelu_lut_32(x)))(x_grad)
    
    print(f"Input: {x_grad}")
    print(f"Exact GELU grad: {grad_exact}")
    print(f"LUT-16 grad:     {grad_lut_16}")
    print(f"LUT-32 grad:     {grad_lut_32}")
