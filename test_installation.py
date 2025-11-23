#!/usr/bin/env python3
"""
Test script to verify PyTorch Lucid installation and basic functionality.
"""

import sys
import torch
import numpy as np
from pytorch_lucid import modelzoo, optvis, recipes


def test_basic_imports():
    """Test that all basic imports work."""
    print("Testing basic imports...")
    try:
        from pytorch_lucid import modelzoo, optvis, recipes, misc
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_model_loading():
    """Test model loading functionality."""
    print("\nTesting model loading...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Try to load a small model first
        model_wrapper = modelzoo.load_model('alexnet', device=device)
        print(f"✓ Successfully loaded {model_wrapper.model.__class__.__name__}")
        
        # Test layer extraction
        layers = model_wrapper.get_layer_names()
        print(f"✓ Found {len(layers)} layers")
        
        conv_layers = model_wrapper.get_conv_layers()
        print(f"✓ Found {len(conv_layers)} convolutional layers")
        
        return True
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        return False


def test_basic_visualization():
    """Test basic visualization functionality."""
    print("\nTesting basic visualization...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load a small model
        model_wrapper = modelzoo.load_model('alexnet', device=device)
        
        # Get a convolutional layer
        conv_layers = model_wrapper.get_conv_layers()
        if not conv_layers:
            print("✗ No convolutional layers found")
            return False
        
        target_layer = conv_layers[0]
        print(f"Using layer: {target_layer}")
        
        # Try a simple visualization
        images = optvis.visualize_channel(
            model_wrapper.model,
            target_layer,
            channel_idx=0,
            num_steps=50,  # Quick test
            image_size=(64, 64),  # Small image for quick test
            device=device,
            verbose=False
        )
        
        if images and len(images) > 0:
            print("✓ Basic visualization successful")
            print(f"  Generated {len(images)} image(s)")
            print(f"  Final image shape: {images[-1].shape}")
            return True
        else:
            print("✗ No images generated")
            return False
            
    except Exception as e:
        print(f"✗ Visualization error: {e}")
        return False


def test_recipes():
    """Test recipe functionality."""
    print("\nTesting recipes...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_wrapper = modelzoo.load_model('alexnet', device=device)
        conv_layers = model_wrapper.get_conv_layers()
        
        if conv_layers:
            # Test feature grid creation
            feature_grid = recipes.create_feature_visualization_grid(
                model_wrapper,
                conv_layers[0],
                num_channels=4,
                grid_size=(2, 2),
                num_steps=50,
                image_size=(64, 64),
                device=device
            )
            
            if feature_grid is not None:
                print("✓ Recipe functionality working")
                print(f"  Feature grid shape: {feature_grid.shape}")
                return True
            else:
                print("✗ Feature grid generation failed")
                return False
        else:
            print("✗ No conv layers for recipe test")
            return False
            
    except Exception as e:
        print(f"✗ Recipe error: {e}")
        return False


def test_system_info():
    """Print system information."""
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"NumPy version: {np.__version__}")


def main():
    """Run all tests."""
    print("PyTorch Lucid Installation Test")
    print("=" * 50)
    
    test_system_info()
    
    tests = [
        test_basic_imports,
        test_model_loading,
        test_basic_visualization,
        test_recipes
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Basic Imports",
        "Model Loading", 
        "Basic Visualization",
        "Recipes"
    ]
    
    passed = sum(results)
    total = len(results)
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! PyTorch Lucid is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())