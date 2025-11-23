#!/usr/bin/env python3
"""
Complete Distill Papers Analysis - Comprehensive Example

This script provides a complete analysis combining all techniques from the three Distill papers:
1. Feature Visualization
2. Building Blocks of Interpretability  
3. Thread: Circuits

It performs a comprehensive analysis of a neural network using all available techniques.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pytorch_lucid import modelzoo
from pytorch_lucid.distill import FeatureVisualization, BuildingBlocks, Circuits
from pytorch_lucid.misc.io import save_image, create_image_grid


def create_comprehensive_analysis(model_name='vgg16', output_dir='distill_complete_analysis'):
    """Perform comprehensive analysis using all Distill paper techniques."""
    
    print("="*80)
    print("COMPLETE DISTILL PAPERS ANALYSIS")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Output Directory: {output_dir}")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading {model_name} model...")
    model_wrapper = modelzoo.load_model(model_name, device=device)
    
    # Get model information
    conv_layers = model_wrapper.get_conv_layers()
    linear_layers = model_wrapper.get_linear_layers()
    
    print(f"Convolutional layers: {len(conv_layers)}")
    print(f"Linear layers: {len(linear_layers)}")
    
    # Initialize all analysis tools
    fv = FeatureVisualization(model_wrapper.model, device)
    bb = BuildingBlocks(model_wrapper.model, device)
    circuits = Circuits(model_wrapper.model, device)
    
    # Analysis results storage
    analysis_results = {
        'model_name': model_name,
        'num_conv_layers': len(conv_layers),
        'num_linear_layers': len(linear_layers),
        'feature_visualization': {},
        'building_blocks': {},
        'circuits': {}
    }
    
    # 1. FEATURE VISUALIZATION ANALYSIS
    print("\n" + "="*60)
    print("1. FEATURE VISUALIZATION ANALYSIS")
    print("="*60)
    
    # Select representative layers for analysis
    analysis_layers = []
    if len(conv_layers) >= 3:
        # Early, middle, and late layers
        analysis_layers = [
            conv_layers[0],           # Early layer
            conv_layers[len(conv_layers)//2],  # Middle layer  
            conv_layers[-1]           # Late layer
        ]
    else:
        analysis_layers = conv_layers[:3]
    
    print(f"Analyzing layers: {analysis_layers}")
    
    for i, layer in enumerate(analysis_layers):
        print(f"\nAnalyzing layer {i+1}/{len(analysis_layers)}: {layer}")
        
        try:
            # Basic channel visualization
            print("  - Basic channel visualization...")
            basic_viz = fv.visualize_channel(layer, 0, num_steps=256, image_size=(128, 128))
            
            # Diversity visualization
            print("  - Diversity analysis...")
            diverse_images = fv.create_diversity_visualization(layer, 0, num_diverse=4, num_steps=128)
            
            # Feature grid
            print("  - Feature grid...")
            layer_info = model_wrapper.get_layer_info(layer)
            if 'out_channels' in layer_info:
                num_channels = min(9, layer_info['out_channels'])
                channel_indices = list(range(num_channels))
            else:
                channel_indices = list(range(9))
            
            feature_grid = fv.create_feature_visualization_grid(
                layer, channel_indices, grid_size=(3, 3), num_steps=128
            )
            
            # Store results
            analysis_results['feature_visualization'][layer] = {
                'basic_visualization': basic_viz is not None,
                'diversity_images': len(diverse_images),
                'feature_grid': feature_grid.shape if hasattr(feature_grid, 'shape') else None
            }
            
            # Save visualizations
            if basic_viz is not None:
                save_image(
                    torch.from_numpy(basic_viz).permute(2, 0, 1).unsqueeze(0),
                    f'{output_dir}/fv_basic_{layer.replace(".", "_")}.png'
                )
            
            if diverse_images:
                diverse_grid = create_image_grid(diverse_images, (2, 2))
                diverse_array = diverse_grid.detach().cpu().numpy().transpose(1, 2, 0)
                save_image(
                    torch.from_numpy(diverse_array).permute(2, 0, 1).unsqueeze(0),
                    f'{output_dir}/fv_diverse_{layer.replace(".", "_")}.png'
                )
            
            if feature_grid is not None:
                save_image(
                    torch.from_numpy(feature_grid).permute(2, 0, 1).unsqueeze(0),
                    f'{output_dir}/fv_grid_{layer.replace(".", "_")}.png'
                )
            
        except Exception as e:
            print(f"  Error analyzing {layer}: {e}")
            analysis_results['feature_visualization'][layer] = {'error': str(e)}
    
    # 2. BUILDING BLOCKS ANALYSIS
    print("\n" + "="*60)
    print("2. BUILDING BLOCKS OF INTERPRETABILITY ANALYSIS")
    print("="*60)
    
    for i, layer in enumerate(analysis_layers):
        print(f"\nAnalyzing layer {i+1}/{len(analysis_layers)}: {layer}")
        
        try:
            # Semantic dictionary
            print("  - Semantic dictionary...")
            semantic_dict = bb.create_semantic_dictionary(layer, top_k=12, num_steps=128)
            
            # Combined interface (using synthetic test image)
            print("  - Combined interface...")
            test_image = np.random.rand(224, 224, 3).astype(np.float32)
            # Create smooth pattern
            x = np.linspace(0, 1, 224)
            y = np.linspace(0, 1, 224)
            X, Y = np.meshgrid(x, y)
            test_image[:, :, 0] = np.sin(X * 4 * np.pi) * np.cos(Y * 4 * np.pi)
            test_image = (test_image - test_image.min()) / (test_image.max() - test_image.min())
            
            interface = bb.create_combined_interface(test_image, layer, top_k=4)
            
            # Store results
            analysis_results['building_blocks'][layer] = {
                'semantic_dictionary': semantic_dict['num_features'] if 'num_features' in semantic_dict else 0,
                'combined_interface': interface is not None,
                'attribution_available': 'attribution_map' in interface and interface['attribution_map'] is not None
            }
            
            # Save visualizations
            if semantic_dict.get('semantic_grid') is not None:
                save_image(
                    torch.from_numpy(semantic_dict['semantic_grid']).permute(2, 0, 1).unsqueeze(0),
                    f'{output_dir}/bb_semantic_{layer.replace(".", "_")}.png'
                )
            
        except Exception as e:
            print(f"  Error analyzing {layer}: {e}")
            analysis_results['building_blocks'][layer] = {'error': str(e)}
    
    # 3. CIRCUITS ANALYSIS
    print("\n" + "="*60)
    print("3. THREAD: CIRCUITS ANALYSIS")
    print("="*60)
    
    for i, layer in enumerate(analysis_layers):
        print(f"\nAnalyzing layer {i+1}/{len(analysis_layers)}: {layer}")
        
        try:
            # Curve detector analysis
            print("  - Curve detector analysis...")
            curve_analysis = circuits.analyze_curve_detectors(layer, num_samples=30)
            
            # Weight banding analysis
            print("  - Weight banding analysis...")
            banding = circuits.analyze_weight_banding(layer, num_bands=4)
            
            # Branch specialization
            print("  - Branch specialization...")
            specialization = circuits.detect_branch_specialization(layer, num_samples=20)
            
            # Weight visualization
            print("  - Weight visualization...")
            weight_viz = circuits.visualize_weights(layer, max_channels=16)
            
            # High-low frequency detectors
            print("  - High-low frequency detectors...")
            hl_analysis = circuits.analyze_high_low_frequency_detectors(layer, num_samples=20)
            
            # Store results
            analysis_results['circuits'][layer] = {
                'curve_detectors': curve_analysis['num_curve_detectors'],
                'weight_bands': len(banding['band_stats']) if 'band_stats' in banding else 0,
                'specialization_groups': specialization['num_groups'] if 'num_groups' in specialization else 0,
                'specialization_score': specialization.get('specialization_score', 0),
                'hl_detectors': hl_analysis['num_hl_detectors'],
                'weight_visualization': weight_viz['num_filters'] if 'num_filters' in weight_viz else 0
            }
            
            # Save visualizations
            if weight_viz.get('weight_grid') is not None:
                save_image(
                    torch.from_numpy(weight_viz['weight_grid']).permute(2, 0, 1).unsqueeze(0),
                    f'{output_dir}/circuits_weights_{layer.replace(".", "_")}.png'
                )
            
            if curve_analysis.get('curve_visualizations'):
                curve_images = list(curve_analysis['curve_visualizations'].values())
                if curve_images:
                    curve_grid = create_image_grid(curve_images, (2, 2))
                    curve_array = curve_grid.detach().cpu().numpy().transpose(1, 2, 0)
                    save_image(
                        torch.from_numpy(curve_array).permute(2, 0, 1).unsqueeze(0),
                        f'{output_dir}/circuits_curves_{layer.replace(".", "_")}.png'
                    )
            
            if hl_analysis.get('hl_visualizations'):
                hl_images = list(hl_analysis['hl_visualizations'].values())
                if hl_images:
                    hl_grid = create_image_grid(hl_images, (2, 2))
                    hl_array = hl_grid.detach().cpu().numpy().transpose(1, 2, 0)
                    save_image(
                        torch.from_numpy(hl_array).permute(2, 0, 1).unsqueeze(0),
                        f'{output_dir}/circuits_hl_{layer.replace(".", "_")}.png'
                    )
            
        except Exception as e:
            print(f"  Error analyzing {layer}: {e}")
            analysis_results['circuits'][layer] = {'error': str(e)}
    
    # 4. COMPREHENSIVE SUMMARY
    print("\n" + "="*80)
    print("4. COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)
    
    create_comprehensive_summary(analysis_results, output_dir)
    
    return analysis_results


def create_comprehensive_summary(results, output_dir):
    """Create a comprehensive summary visualization."""
    
    print("Creating comprehensive summary visualization...")
    
    # Create summary figure
    fig = plt.figure(figsize=(20, 24))
    
    # Title
    fig.suptitle(f'Complete Distill Papers Analysis - {results["model_name"]}', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # 1. Model Architecture Overview
    ax1 = fig.add_subplot(4, 2, 1)
    layers_data = [results['num_conv_layers'], results['num_linear_layers']]
    colors = ['skyblue', 'lightcoral']
    labels = ['Conv Layers', 'Linear Layers']
    
    bars = ax1.bar(labels, layers_data, color=colors, alpha=0.8)
    ax1.set_title('Model Architecture')
    ax1.set_ylabel('Number of Layers')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Feature Visualization Results
    ax2 = fig.add_subplot(4, 2, 2)
    fv_success = sum(1 for layer in results['feature_visualization'].values() 
                     if isinstance(layer, dict) and 'error' not in layer)
    fv_total = len(results['feature_visualization'])
    
    fv_data = [fv_success, fv_total - fv_success]
    fv_labels = ['Successful', 'Failed']
    fv_colors = ['green', 'red']
    
    wedges, texts, autotexts = ax2.pie(fv_data, labels=fv_labels, colors=fv_colors, 
                                       autopct='%1.1f%%', startangle=90)
    ax2.set_title('Feature Visualization Success Rate')
    
    # 3. Building Blocks Results
    ax3 = fig.add_subplot(4, 2, 3)
    bb_data = []
    bb_labels = []
    
    for layer_name, layer_results in results['building_blocks'].items():
        if isinstance(layer_results, dict) and 'error' not in layer_results:
            bb_data.append(layer_results.get('semantic_dictionary', 0))
            bb_labels.append(layer_name.split('.')[-1])
    
    if bb_data:
        bars = ax3.bar(bb_labels, bb_data, color='orange', alpha=0.8)
        ax3.set_title('Building Blocks - Semantic Dictionary Features')
        ax3.set_ylabel('Number of Features')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Circuits Analysis Results
    ax4 = fig.add_subplot(4, 2, 4)
    
    # Circuit metrics heatmap
    circuit_metrics = []
    circuit_labels = []
    
    for layer_name, layer_results in results['circuits'].items():
        if isinstance(layer_results, dict) and 'error' not in layer_results:
            circuit_labels.append(layer_name.split('.')[-1])
            circuit_metrics.append([
                layer_results.get('curve_detectors', 0),
                layer_results.get('hl_detectors', 0),
                layer_results.get('weight_bands', 0),
                layer_results.get('specialization_groups', 0)
            ])
    
    if circuit_metrics:
        circuit_metrics = np.array(circuit_metrics)
        im = ax4.imshow(circuit_metrics.T, cmap='Blues', aspect='auto')
        ax4.set_title('Circuits Analysis Heatmap')
        ax4.set_xlabel('Layers')
        ax4.set_ylabel('Metrics')
        ax4.set_yticks(range(4))
        ax4.set_yticklabels(['Curve Detectors', 'HL Detectors', 'Weight Bands', 'Spec Groups'])
        ax4.set_xticks(range(len(circuit_labels)))
        ax4.set_xticklabels(circuit_labels, rotation=45, ha='right')
        
        # Add colorbar
        plt.colorbar(im, ax=ax4)
    
    # 5. Layer-wise Feature Visualization Summary
    ax5 = fig.add_subplot(4, 2, 5)
    
    # Feature visualization success by layer
    fv_success_by_layer = []
    fv_layer_names = []
    
    for layer_name, layer_results in results['feature_visualization'].items():
        if isinstance(layer_results, dict) and 'error' not in layer_results:
            fv_layer_names.append(layer_name.split('.')[-1])
            fv_success_by_layer.append(1)  # Success
        else:
            fv_layer_names.append(layer_name.split('.')[-1])
            fv_success_by_layer.append(0)  # Failure
    
    if fv_success_by_layer:
        colors = ['green' if x == 1 else 'red' for x in fv_success_by_layer]
        bars = ax5.bar(fv_layer_names, fv_success_by_layer, color=colors, alpha=0.8)
        ax5.set_title('Feature Visualization Success by Layer')
        ax5.set_ylabel('Success (1) / Failure (0)')
        ax5.set_ylim(-0.1, 1.1)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # 6. Circuits Specialization Analysis
    ax6 = fig.add_subplot(4, 2, 6)
    
    specialization_scores = []
    specialization_layers = []
    
    for layer_name, layer_results in results['circuits'].items():
        if isinstance(layer_results, dict) and 'error' not in layer_results:
            if 'specialization_score' in layer_results:
                specialization_scores.append(layer_results['specialization_score'])
                specialization_layers.append(layer_name.split('.')[-1])
    
    if specialization_scores:
        bars = ax6.bar(specialization_layers, specialization_scores, color='purple', alpha=0.8)
        ax6.set_title('Branch Specialization Scores')
        ax6.set_ylabel('Specialization Score')
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
    
    # 7. Comprehensive Statistics Table
    ax7 = fig.add_subplot(4, 2, (7, 8))
    ax7.axis('off')
    
    # Create statistics table
    stats_text = f"""
COMPREHENSIVE ANALYSIS RESULTS

Model: {results['model_name']}
Total Layers: {results['num_conv_layers']} Conv + {results['num_linear_layers']} Linear

FEATURE VISUALIZATION:
- Layers Analyzed: {len(results['feature_visualization'])}
- Successful: {sum(1 for r in results['feature_visualization'].values() if isinstance(r, dict) and 'error' not in r)}
- Failed: {sum(1 for r in results['feature_visualization'].values() if isinstance(r, dict) and 'error' in r)}

BUILDING BLOCKS:
- Semantic Dictionaries: {sum(r.get('semantic_dictionary', 0) for r in results['building_blocks'].values() if isinstance(r, dict) and 'error' not in r)}
- Combined Interfaces: {sum(1 for r in results['building_blocks'].values() if isinstance(r, dict) and r.get('combined_interface', False))}

CIRCUITS ANALYSIS:
- Total Curve Detectors: {sum(r.get('curve_detectors', 0) for r in results['circuits'].values() if isinstance(r, dict))}
- Total HL Detectors: {sum(r.get('hl_detectors', 0) for r in results['circuits'].values() if isinstance(r, dict))}
- Total Weight Bands: {sum(r.get('weight_bands', 0) for r in results['circuits'].values() if isinstance(r, dict))}
- Total Specialization Groups: {sum(r.get('specialization_groups', 0) for r in results['circuits'].values() if isinstance(r, dict))}

TECHNIQUES IMPLEMENTED:
✓ Feature Visualization (all techniques)
✓ Building Blocks (semantic dictionaries, combined interfaces)
✓ Circuits (curve detectors, weight analysis, banding, specialization)
✓ Advanced regularization and optimization
✓ Comprehensive visualization and analysis
"""
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Summary visualization saved to: {output_dir}/comprehensive_summary.png")


def main():
    """Run complete Distill papers analysis."""
    
    print("Complete Distill Papers Analysis")
    print("=" * 80)
    print("This script performs comprehensive analysis using techniques from:")
    print("1. Feature Visualization (Olah et al., 2017)")
    print("2. Building Blocks of Interpretability (Olah et al., 2018)")
    print("3. Thread: Circuits (Cammarata et al., 2020)")
    print()
    
    # Run analysis
    results = create_comprehensive_analysis(model_name='vgg16')
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\nSuccessfully analyzed {results['model_name']} using all Distill paper techniques:")
    print(f"- Model Architecture: {results['num_conv_layers']} conv + {results['num_linear_layers']} linear layers")
    
    # Feature Visualization Summary
    fv_success = sum(1 for r in results['feature_visualization'].values() if isinstance(r, dict) and 'error' not in r)
    fv_total = len(results['feature_visualization'])
    print(f"- Feature Visualization: {fv_success}/{fv_total} layers successful")
    
    # Building Blocks Summary
    bb_features = sum(r.get('semantic_dictionary', 0) for r in results['building_blocks'].values() if isinstance(r, dict) and 'error' not in r)
    print(f"- Building Blocks: {bb_features} features in semantic dictionaries")
    
    # Circuits Summary
    total_curves = sum(r.get('curve_detectors', 0) for r in results['circuits'].values() if isinstance(r, dict))
    total_hl = sum(r.get('hl_detectors', 0) for r in results['circuits'].values() if isinstance(r, dict))
    total_bands = sum(r.get('weight_bands', 0) for r in results['circuits'].values() if isinstance(r, dict))
    print(f"- Circuits Analysis: {total_curves} curve detectors, {total_hl} HL detectors, {total_bands} weight bands")
    
    print(f"\nAll results saved in: distill_complete_analysis/")
    print("Ready for research publication and further exploration!")


if __name__ == "__main__":
    main()