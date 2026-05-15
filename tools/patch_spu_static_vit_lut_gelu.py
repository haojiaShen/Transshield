#!/usr/bin/env python3
"""
Patch spu_static_vit.py to support LUT GELU activation.

This script modifies the spu_static_vit.py file to add support for
piecewise linear GELU activation (LUT GELU).
"""

import sys
sys.path.insert(0, '/home/yclcg/Transshield_final')

import os


def patch_spu_static_vit():
    """Patch spu_static_vit.py to support LUT GELU activation."""
    
    # Path to the file
    file_path = "integrations/openbumblebee/e2e_secure_vit/spu_static_vit.py"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "lut_gelu" in content:
        print("File already patched with LUT GELU support")
        return True
    
    # Find the activate function
    activate_function_start = content.find("def activate(x, alpha, beta):")
    if activate_function_start == -1:
        print("Could not find activate function")
        return False
    
    # Find the end of the activate function
    # Look for the next function definition or the end of the file
    activate_function_end = content.find("\ndef ", activate_function_start + 1)
    if activate_function_end == -1:
        activate_function_end = len(content)
    
    # Extract the activate function
    activate_function = content[activate_function_start:activate_function_end]
    
    # Create the new activate function with LUT GELU support
    new_activate_function = '''def activate(x, alpha, beta):
        if activation_clip_value > 0.0:
            x = jnp.clip(x, -activation_clip_value, activation_clip_value)
        if activation_kind == "gelu":
            return gelu_exact(x)
        if activation_kind in {"fixed_square", "learnable_square"}:
            return alpha * (x * x)
        if activation_kind in {"learnable_quadratic", "learnable_quadratic_gelu_init"}:
            return alpha * (x * x) + beta * x
        if activation_kind == "lut_gelu_16":
            # LUT GELU with 16 segments (piecewise linear approximation)
            breakpoints = jnp.linspace(-8.0, 8.0, 17)
            values = 0.5 * breakpoints * (1.0 + jsp_special.erf(breakpoints / jnp.sqrt(2.0)))
            return jnp.interp(x, breakpoints, values)
        if activation_kind == "lut_gelu_32":
            # LUT GELU with 32 segments (piecewise linear approximation)
            breakpoints = jnp.linspace(-8.0, 8.0, 33)
            values = 0.5 * breakpoints * (1.0 + jsp_special.erf(breakpoints / jnp.sqrt(2.0)))
            return jnp.interp(x, breakpoints, values)
        raise ValueError(f"unsupported activation kind: {activation_kind}")
'''
    
    # Replace the activate function
    new_content = content[:activate_function_start] + new_activate_function + content[activate_function_end:]
    
    # Write the modified file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Successfully patched {file_path}")
    print("Added support for:")
    print("  - lut_gelu_16: LUT GELU with 16 segments")
    print("  - lut_gelu_32: LUT GELU with 32 segments")
    
    return True


if __name__ == "__main__":
    success = patch_spu_static_vit()
    if success:
        print("\nPatch applied successfully!")
    else:
        print("\nFailed to apply patch")
