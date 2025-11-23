# PyTorch Lucid

A PyTorch implementation of the Lucid library for neural network interpretability and feature visualization.

## Overview

PyTorch Lucid is a collection of infrastructure and tools for research in neural network interpretability, ported from the original TensorFlow Lucid library. It provides tools for:

- **Feature Visualization**: Generate images that maximally activate specific neurons, channels, or layers
- **Style Transfer**: Transfer artistic styles between images using neural networks
- **DeepDream**: Create dream-like images by maximizing layer activations
- **Model Analysis**: Understand what different parts of a neural network are learning

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- pillow
- tqdm

### Install from source

```bash
git clone https://github.com/your-username/pytorch-lucid.git
cd pytorch-lucid
pip install -e .
```

### Install dependencies

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

## Quick Start

### Basic Feature Visualization

```python
import torch
from pytorch_lucid import modelzoo, optvis

# Load a pre-trained model
model = modelzoo.load_model('vgg16')

# Visualize a specific channel
images = optvis.visualize_channel(
    model.model, 
    layer_name='features.10', 
    channel_idx=5,
    num_steps=512
)

# Display the result
from pytorch_lucid.misc.io import show_image
show_image(images[-1])
```

### Style Transfer

```python
from pytorch_lucid.optvis.style import create_style_objective

# Load content and style images
content_img = load_image('content.jpg')
style_img = load_image('style.jpg')

# Create style transfer objective
style_obj = create_style_objective(
    style_img, content_img, model.model, 
    layer_names=['features.1', 'features.6', 'features.11']
)

# Perform style transfer
result = optvis.render_vis(model.model, style_obj, num_steps=300)
```

## Core Components

### 1. Model Zoo (`pytorch_lucid.modelzoo`)

Utilities for loading and working with pre-trained models:

```python
from pytorch_lucid import modelzoo

# Load popular models
model = modelzoo.load_model('vgg16')
model = modelzoo.load_model('resnet50')
model = modelzoo.load_model('alexnet')

# Get model information
layers = model.get_layer_names()
conv_layers = model.get_conv_layers()
```

### 2. Optimization-based Visualization (`pytorch_lucid.optvis`)

Core visualization framework:

```python
from pytorch_lucid import optvis

# Visualize neurons, channels, or layers
images = optvis.visualize_neuron(model, layer, neuron_idx)
images = optvis.visualize_channel(model, layer, channel_idx)
images = optvis.visualize_layer(model, layer)

# Custom objectives
def my_objective(activations):
    return activations['layer_name'][:, 0].mean()

images = optvis.render_vis(model, my_objective)
```

### 3. Objectives (`pytorch_lucid.optvis.objectives`)

Predefined objective functions:

```python
from pytorch_lucid.optvis.objectives import channel, neuron, layer, deepdream

# Channel objective
obj = channel('conv1', 5)

# Neuron objective  
obj = neuron('conv2', 10, x=10, y=10)

# Combined objectives
obj = 0.5 * channel('conv1', 5) + 0.5 * channel('conv1', 10)
```

### 4. Parameterizations (`pytorch_lucid.optvis.param`)

Different ways to parameterize images:

```python
from pytorch_lucid.optvis.param import image, fft_image

# Direct pixel parameterization
param = image(shape=(1, 3, 224, 224))

# Fourier space parameterization (often better for optimization)
param = fft_image(shape=(1, 3, 224, 224))
```

### 5. Transforms (`pytorch_lucid.optvis.transform`)

Image transformations for robust optimization:

```python
from pytorch_lucid.optvis.transform import standard_transforms, Jitter, Scale

# Standard transforms for feature visualization
transforms = standard_transforms(jitter=16, scale=(0.9, 1.1))

# Custom transforms
transforms = Jitter(16) + Scale((0.9, 1.1)) + Rotate((-5, 5))
```

## Advanced Usage

### Custom Visualization Pipeline

```python
from pytorch_lucid import optvis

def custom_visualization(model, layer_name, channel_idx):
    # Create custom objective
    obj = optvis.objectives.channel(layer_name, channel_idx)
    
    # Add regularization
    obj = obj - 0.1 * optvis.objectives.total_variation()
    
    # Custom parameterization
    param = optvis.param.fft_image((1, 3, 224, 224))
    
    # Custom transforms
    transforms = optvis.transform.standard_transforms(
        jitter=8, scale=(0.95, 1.05), rotate=(-3, 3)
    )
    
    # Optimize
    images = optvis.render_vis(
        model, obj, param_f=param, transforms=transforms,
        thresholds=(128, 256, 512)
    )
    
    return images
```

### Model Analysis

```python
from pytorch_lucid import modelzoo

# Load model
model = modelzoo.load_model('resnet50')

# Get feature layers (good for visualization)
feature_layers = modelzoo.get_feature_layers(model)

# Analyze layer representations
results = {}
for layer in feature_layers[:5]:  # First 5 feature layers
    visualizations = optvis.visualize_channel(model.model, layer, 0)
    results[layer] = visualizations[-1]
```

## Examples

The `examples/` directory contains comprehensive examples:

- `basic_visualization_example.py`: Basic neuron, channel, and layer visualizations
- `style_transfer_example.py`: Neural style transfer with different configurations
- `deepdream_example.py`: DeepDream-style visualizations
- `model_analysis_example.py`: Analyzing model representations across layers

Run an example:

```bash
python examples/basic_visualization_example.py
```

## Recipes

The `recipes/` module provides ready-to-use functions for common tasks:

```python
from pytorch_lucid import recipes

# Visualize multiple neurons
neuron_images = recipes.visualize_neurons(model, layer, [0, 1, 2, 3])

# Create feature grid
grid = recipes.create_feature_visualization_grid(model, layer, num_channels=9)

# Compare layer representations
comparisons = recipes.compare_layer_representations(model, layer_names)
```

## Best Practices

### 1. Choose Appropriate Layers

- **Early layers**: Simple features like edges, colors, textures
- **Middle layers**: More complex patterns, shapes, textures
- **Later layers**: High-level concepts, objects, semantic features

### 2. Parameterization Choice

- **Direct pixels**: Good for simple visualizations
- **FFT parameterization**: Better for natural-looking images, smoother optimization

### 3. Transformation Strategy

- Use transforms for more robust visualizations
- Standard transforms: jitter, scale, rotation, pad/crop
- Adjust transform strength based on desired effect

### 4. Optimization Parameters

- **Learning rate**: Start with 0.05, adjust as needed
- **Steps**: 256-512 for quick results, 1024+ for high quality
- **Regularization**: Add total variation or other regularizers for smoother results

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce image size or batch size
2. **No visualization**: Check layer names, ensure model is in eval mode
3. **Poor quality**: Increase optimization steps, adjust transforms
4. **NaN values**: Reduce learning rate, check for proper normalization

### Performance Tips

- Use GPU for faster optimization
- Use FFT parameterization for better convergence
- Apply appropriate regularization
- Choose meaningful layers for visualization

## Comparison with TensorFlow Lucid

This PyTorch implementation aims to provide similar functionality to the original TensorFlow Lucid:

| Feature | TensorFlow Lucid | PyTorch Lucid |
|---------|------------------|---------------|
| Core visualization | ✅ | ✅ |
| Style transfer | ✅ | ✅ |
| DeepDream | ✅ | ✅ |
| Model zoo | ✅ | ✅ |
| Custom objectives | ✅ | ✅ |
| FFT parameterization | ✅ | ✅ |
| Transforms | ✅ | ✅ |

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

This project is licensed under the Apache License 2.0, same as the original Lucid library.

## Acknowledgments

This project is a PyTorch port of the original TensorFlow Lucid library by the TensorFlow team. We thank the original authors for their excellent work in neural network interpretability.

## Citation

If you use this library in your research, please cite:

```bibtex
@misc{pytorch-lucid,
  title={PyTorch Lucid: Neural Network Visualization in PyTorch},
  author={PyTorch Lucid Contributors},
  year={2024},
  url={https://github.com/your-username/pytorch-lucid}
}
```

And also cite the original Lucid paper:

```bibtex
@article{lucid,
  title={The Building Blocks of Interpretability},
  author={Olah, Chris and Satyanarayan, Arvind and Johnson, Ian and Carter, Shan and Schubert, Ludwig and Ye, Katherine and Mordvintsev, Alexander},
  journal={Distill},
  year={2018},
  url={http://distill.pub/2018/building-blocks}
}
```