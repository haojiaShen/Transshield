#!/usr/bin/env python3
"""
Patch dyvit.py to support LUT GELU activation (fix both get_act_layer functions).

This script modifies the dyvit.py file to add support for
piecewise linear GELU activation (LUT GELU).
"""

import sys
sys.path.insert(0, '/home/yclcg/Transshield_final')

import os


def patch_dyvit():
    """Patch dyvit.py to support LUT GELU activation."""
    
    # Path to the file
    file_path = "models/dyvit.py"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if content.count("lut_gelu_16") > 2:
        print("File already patched with LUT GELU support")
        return True
    
    # Find all get_act_layer functions
    import re
    pattern = r'def get_act_layer\(act_layer\):.*?(?=\ndef |\nclass |\Z)'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    print(f"Found {len(matches)} get_act_layer functions")
    
    if len(matches) < 2:
        print("Could not find both get_act_layer functions")
        return False
    
    # Patch both functions
    for i, match in enumerate(matches):
        start = match.start()
        end = match.end()
        
        # Get the original function
        original = content[start:end]
        
        # Check if it already has lut_gelu support
        if "lut_gelu" in original:
            print(f"Function {i+1} already patched")
            continue
        
        # Add lut_gelu support
        # Find the last "raise ValueError" line
        last_raise = original.rfind("raise ValueError")
        if last_raise == -1:
            print(f"Could not find raise ValueError in function {i+1}")
            continue
        
        # Insert lut_gelu support before the raise ValueError
        new_function = original[:last_raise] + '''        if act_layer == 'lut_gelu_16':
            return LUTGELU16
        if act_layer == 'lut_gelu_32':
            return LUTGELU32
        ''' + original[last_raise:]
        
        # Replace the function
        content = content[:start] + new_function + content[end:]
        
        print(f"Patched function {i+1}")
    
    # Write the modified file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully patched {file_path}")
    print("Added support for:")
    print("  - lut_gelu_16: LUT GELU with 16 segments")
    print("  - lut_gelu_32: LUT GELU with 32 segments")
    
    return True


if __name__ == "__main__":
    success = patch_dyvit()
    if success:
        print("\nPatch applied successfully!")
    else:
        print("\nFailed to apply patch")
