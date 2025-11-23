#!/usr/bin/env python3
"""
Style transfer example for PyTorch Lucid.

This example demonstrates how to:
1. Load content and style images
2. Create style transfer objectives
3. Perform neural style transfer
4. Save and display the results
"""

import torch
import numpy as np
from pytorch_lucid import modelzoo, optvis, misc
import matplotlib.pyplot as plt
from PIL import Image


def main():
    """Run the style transfer example."""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a pre-trained model for style transfer
    print("\nLoading pre-trained VGG19 model...")
    model_wrapper = modelzoo.load_model('vgg19', device=device)
    
    # Create dummy content and style images for demonstration
    print("\nCreating example content and style images...")
    
    # Create a simple content image (geometric shapes)
    content_array = np.zeros((256, 256, 3), dtype=np.uint8)
    # Add a white circle
    center = (128, 128)
    radius = 60
    y, x = np.ogrid[:256, :256]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    content_array[mask] = [255, 255, 255]
    # Add a rectangle
    content_array[50:100, 50:200] = [255, 0, 0]  # Red rectangle
    content_array[150:200, 50:200] = [0, 255, 0]  # Green rectangle
    
    content_image = Image.fromarray(content_array)
    content_image.save('content_image.png')
    
    # Create a style image (colorful patterns)
    style_array = np.zeros((256, 256, 3), dtype=np.uint8)
    # Create colorful wave pattern
    x = np.linspace(0, 4*np.pi, 256)
    y = np.linspace(0, 4*np.pi, 256)
    X, Y = np.meshgrid(x, y)
    
    # Colorful sine wave pattern
    style_array[:, :, 0] = (np.sin(X) * 127 + 128).astype(np.uint8)  # Red
    style_array[:, :, 1] = (np.cos(Y) * 127 + 128).astype(np.uint8)  # Green
    style_array[:, :, 2] = (np.sin(X + Y) * 127 + 128).astype(np.uint8)  # Blue
    
    style_image = Image.fromarray(style_array)
    style_image.save('style_image.png')
    
    # Convert images to tensors
    content_tensor = misc.io.load_image('content_image.png', size=(224, 224), device=device)
    style_tensor = misc.io.load_image('style_image.png', size=(224, 224), device=device)
    
    # Display input images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Content image
    content_display = content_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(content_display)
    axes[0].set_title('Content Image')
    axes[0].axis('off')
    
    # Style image
    style_display = style_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    axes[1].imshow(style_display)
    axes[1].set_title('Style Image')
    axes[1].axis('off')
    
    plt.suptitle('Input Images for Style Transfer')
    plt.tight_layout()
    plt.savefig('style_transfer_inputs.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 1: Basic Style Transfer
    print("\n" + "="*50)
    print("Example 1: Basic Style Transfer")
    print("="*50)
    
    # Create style transfer objective
    style_layers = {
        'features.1': 1.0,   # conv1_1
        'features.6': 0.75,  # conv2_1
        'features.11': 0.2,  # conv3_1
        'features.20': 0.2,  # conv4_1
        'features.29': 0.2   # conv5_1
    }
    
    content_layers = {
        'features.22': 1.0  # conv4_2
    }
    
    from pytorch_lucid.optvis.style import StyleTransferObjective
    
    style_objective = StyleTransferObjective(
        style_weight=1e6,
        content_weight=1.0,
        style_layers=style_layers,
        content_layers=content_layers
    )
    
    # Extract activations for style and content
    print("Extracting style and content activations...")
    
    # Extract style activations
    style_activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                style_activations[name] = output[0]
            else:
                style_activations[name] = output
        return hook
    
    # Register hooks
    for layer_name in style_layers.keys():
        try:
            module = dict(model_wrapper.model.named_modules())[layer_name]
            hook = module.register_forward_hook(get_activation(layer_name))
            hooks.append(hook)
        except KeyError:
            continue
    
    # Forward pass for style
    with torch.no_grad():
        _ = model_wrapper.model(style_tensor)
    style_objective.set_style_target(style_activations)
    
    # Clear activations
    style_activations.clear()
    
    # Forward pass for content
    with torch.no_grad():
        _ = model_wrapper.model(content_tensor)
    style_objective.set_content_target(style_activations)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Perform style transfer optimization
    print("Performing style transfer optimization...")
    
    # Create parameterization
    param_f = optvis.param.image((1, 3, 224, 224), decorrelate=True, sigmoid=True)
    
    # Create optimizer
    optimizer = torch.optim.Adam(param_f.parameters(), lr=0.05)
    
    # Perform optimization
    num_steps = 300
    results = optvis.render_vis(
        model_wrapper.model,
        style_objective,
        param_f=param_f,
        optimizer=optimizer,
        thresholds=(num_steps,),
        device=device,
        verbose=True
    )
    
    if results:
        result_image = results[-1]
        
        # Display result
        plt.figure(figsize=(10, 10))
        plt.imshow(result_image)
        plt.title('Style Transfer Result')
        plt.axis('off')
        plt.savefig('style_transfer_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save result
        misc.io.save_image(torch.from_numpy(result_image).permute(2, 0, 1).unsqueeze(0),
                          'style_transfer_output.png')
    
    # Example 2: Different Style Weights
    print("\n" + "="*50)
    print("Example 2: Different Style Weights")
    print("="*50)
    
    style_weights = [1e4, 1e5, 1e6, 1e7]  # Different style emphasis
    
    fig, axes = plt.subplots(1, len(style_weights), figsize=(16, 4))
    
    for i, style_weight in enumerate(style_weights):
        print(f"Style weight: {style_weight}")
        
        # Create new objective with different weight
        style_obj = StyleTransferObjective(
            style_weight=style_weight,
            content_weight=1.0,
            style_layers=style_layers,
            content_layers=content_layers
        )
        
        # Set targets (reuse from before)
        style_obj.set_style_target(style_activations)
        style_obj.set_content_target(style_activations)
        
        # New parameterization
        param_f_new = optvis.param.image((1, 3, 224, 224), decorrelate=True, sigmoid=True)
        optimizer_new = torch.optim.Adam(param_f_new.parameters(), lr=0.05)
        
        # Optimize
        results_new = optvis.render_vis(
            model_wrapper.model,
            style_obj,
            param_f=param_f_new,
            optimizer=optimizer_new,
            thresholds=(200,),
            device=device,
            verbose=False
        )
        
        if results_new:
            axes[i].imshow(results_new[-1])
            axes[i].set_title(f'Style Weight: {style_weight:.0e}')
            axes[i].axis('off')
    
    plt.suptitle('Style Transfer with Different Weights')
    plt.tight_layout()
    plt.savefig('style_transfer_weights.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Example 3: Multi-layer Style Transfer
    print("\n" + "="*50)
    print("Example 3: Multi-layer Style Analysis")
    print("="*50)
    
    # Visualize different style layers individually
    individual_style_layers = ['features.1', 'features.6', 'features.11', 'features.20']
    
    fig, axes = plt.subplots(1, len(individual_style_layers), figsize=(16, 4))
    
    for i, style_layer in enumerate(individual_style_layers):
        print(f"Style from layer: {style_layer}")
        
        # Single layer style objective
        single_style_layers = {style_layer: 1.0}
        style_obj_single = StyleTransferObjective(
            style_weight=1e6,
            content_weight=0.1,  # Less content emphasis
            style_layers=single_style_layers,
            content_layers=content_layers
        )
        
        # Extract activations for single style layer
        style_activations_single = {}
        hooks_single = []
        
        try:
            module = dict(model_wrapper.model.named_modules())[style_layer]
            hook = module.register_forward_hook(get_activation(style_layer))
            hooks_single.append(hook)
            
            with torch.no_grad():
                _ = model_wrapper.model(style_tensor)
            style_obj_single.set_style_target(style_activations_single)
            
            # Clean up
            for hook in hooks_single:
                hook.remove()
            style_activations_single.clear()
            
            # Forward pass for content
            with torch.no_grad():
                _ = model_wrapper.model(content_tensor)
            style_obj_single.set_content_target(style_activations_single)
            
            # Optimize
            param_f_single = optvis.param.image((1, 3, 224, 224), decorrelate=True, sigmoid=True)
            optimizer_single = torch.optim.Adam(param_f_single.parameters(), lr=0.05)
            
            results_single = optvis.render_vis(
                model_wrapper.model,
                style_obj_single,
                param_f=param_f_single,
                optimizer=optimizer_single,
                thresholds=(200,),
                device=device,
                verbose=False
            )
            
            if results_single:
                axes[i].imshow(results_single[-1])
                axes[i].set_title(f'Style from {style_layer}')
                axes[i].axis('off')
                
        except KeyError:
            print(f"Layer {style_layer} not found")
            axes[i].set_title(f'Layer {style_layer} not found')
            axes[i].axis('off')
    
    plt.suptitle('Style Transfer from Individual Layers')
    plt.tight_layout()
    plt.savefig('style_transfer_layers.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("Style Transfer Complete!")
    print("="*50)
    print("Generated files:")
    print("- style_transfer_inputs.png")
    print("- style_transfer_result.png")
    print("- style_transfer_output.png")
    print("- style_transfer_weights.png")
    print("- style_transfer_layers.png")


if __name__ == "__main__":
    main()