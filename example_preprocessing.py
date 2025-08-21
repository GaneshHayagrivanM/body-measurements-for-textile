#!/usr/bin/env python3
"""
Example script demonstrating video preprocessing capabilities.

This script shows how to use the video preprocessing pipeline
for enhanced body detection in the textile measurement system.
"""

import cv2
import numpy as np
from video_preprocessor import VideoPreprocessor, PreprocessingConfig

def create_example_config():
    """Create example configurations for different scenarios."""
    
    configs = {
        "high_quality": PreprocessingConfig(
            # Minimal preprocessing for high-quality videos
            enable_quality_assessment=True,
            enable_lighting_analysis=True
        ),
        
        "standard": PreprocessingConfig(
            # Standard preprocessing for typical videos
            enable_brightness_contrast=True,
            brightness_factor=1.2,
            contrast_factor=1.1,
            enable_noise_reduction=True,
            noise_reduction_method="bilateral",
            enable_sharpening=True,
            sharpening_strength=0.3,
            enable_quality_assessment=True,
            enable_motion_blur_detection=True
        ),
        
        "challenging": PreprocessingConfig(
            # Full preprocessing for challenging conditions
            enable_brightness_contrast=True,
            brightness_factor=1.3,
            contrast_factor=1.2,
            enable_noise_reduction=True,
            noise_reduction_method="bilateral",
            enable_sharpening=True,
            sharpening_strength=0.5,
            enable_histogram_equalization=True,
            hist_eq_method="adaptive",
            enable_background_subtraction=True,
            background_method="mog2",
            enable_edge_enhancement=True,
            edge_method="canny",
            enable_color_space_conversion=True,
            target_color_space="hsv",
            enable_skin_tone_enhancement=True,
            enable_frame_alignment=True,
            enable_quality_assessment=True,
            enable_motion_blur_detection=True,
            enable_lighting_analysis=True
        ),
        
        "debug": PreprocessingConfig(
            # Configuration for debugging and analysis
            enable_brightness_contrast=True,
            enable_noise_reduction=True,
            enable_quality_assessment=True,
            enable_motion_blur_detection=True,
            enable_lighting_analysis=True,
            enable_visualization=True,
            save_debug_frames=True,
            debug_output_dir="./preprocessing_debug"
        )
    }
    
    return configs

def demonstrate_preprocessing():
    """Demonstrate preprocessing on a synthetic frame."""
    print("Video Preprocessing Demonstration")
    print("=" * 50)
    
    # Create a test frame (simulating a noisy, low-contrast video frame)
    frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
    
    # Add a person-like shape
    cv2.ellipse(frame, (320, 300), (80, 160), 0, 0, 360, (120, 90, 70), -1)
    cv2.circle(frame, (320, 180), 35, (140, 110, 90), -1)
    
    # Add some noise to simulate challenging conditions
    noise = np.random.randint(-30, 30, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    configs = create_example_config()
    
    for config_name, config in configs.items():
        if config_name == "debug":
            continue  # Skip debug config in demonstration
            
        print(f"\nTesting Configuration: {config_name.upper()}")
        print("-" * 30)
        
        preprocessor = VideoPreprocessor(config)
        
        try:
            processed_frame, metrics = preprocessor.process_frame(frame)
            
            print("✓ Frame processed successfully")
            print(f"Processing time: {metrics.get('processing_time', 0):.4f}s")
            
            # Show key quality metrics
            quality_metrics = {
                'overall_quality': 'Overall Quality Score',
                'sharpness': 'Sharpness',
                'brightness': 'Brightness',
                'contrast': 'Contrast'
            }
            
            for key, description in quality_metrics.items():
                if key in metrics:
                    print(f"{description}: {metrics[key]:.3f}")
            
            if 'motion_blur' in metrics:
                print(f"Motion Blur Score: {metrics['motion_blur']:.3f}")
                
        except Exception as e:
            print(f"✗ Processing failed: {e}")
        finally:
            preprocessor.cleanup()

def show_usage_examples():
    """Show command-line usage examples."""
    print("\n" + "=" * 50)
    print("COMMAND LINE USAGE EXAMPLES")
    print("=" * 50)
    
    examples = [
        {
            "title": "Basic Usage (No Preprocessing)",
            "command": "python main.py --video video.mp4 --radius 150 --height 175"
        },
        {
            "title": "Enable All Preprocessing",
            "command": "python main.py --video video.mp4 --radius 150 --height 175 --enable-all-preprocessing"
        },
        {
            "title": "Standard Quality Enhancement",
            "command": """python main.py --video video.mp4 --radius 150 --height 175 \\
  --enable-brightness-contrast --brightness-factor 1.2 \\
  --enable-noise-reduction --noise-reduction-method bilateral \\
  --enable-sharpening --sharpening-strength 0.3 \\
  --enable-quality-assessment"""
        },
        {
            "title": "Challenging Conditions (Full Processing)",
            "command": """python main.py --video video.mp4 --radius 150 --height 175 \\
  --enable-brightness-contrast --brightness-factor 1.3 \\
  --enable-noise-reduction --noise-reduction-method bilateral \\
  --enable-histogram-equalization --hist-eq-method adaptive \\
  --enable-background-subtraction --background-method mog2 \\
  --enable-edge-enhancement --edge-method canny \\
  --enable-color-space-conversion --target-color-space hsv \\
  --enable-skin-tone-enhancement \\
  --enable-frame-alignment \\
  --enable-quality-assessment"""
        },
        {
            "title": "Debug Mode with Visualization",
            "command": """python main.py --video video.mp4 --radius 150 --height 175 \\
  --enable-all-preprocessing \\
  --enable-visualization \\
  --save-debug-frames \\
  --debug-output-dir ./debug_output"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(f"   {example['command']}")

def show_preprocessing_features():
    """Show available preprocessing features."""
    print("\n" + "=" * 50)
    print("AVAILABLE PREPROCESSING FEATURES")
    print("=" * 50)
    
    features = {
        "Frame Enhancement": [
            "Brightness & Contrast Adjustment",
            "Noise Reduction (Bilateral, Gaussian, Median)",
            "Sharpening Filters",
            "Histogram Equalization (Adaptive, Standard)"
        ],
        "Background Processing": [
            "Background Subtraction (MOG2, KNN)",
            "Edge Enhancement (Canny, Sobel, Laplacian)",
            "Adaptive Background Modeling"
        ],
        "Color Space Optimization": [
            "Color Space Conversion (HSV, LAB, RGB)",
            "Skin Tone Enhancement"
        ],
        "Video Stabilization": [
            "Frame Alignment",
            "Video Stabilization (computationally intensive)"
        ],
        "Quality Assessment": [
            "Frame Quality Scoring",
            "Motion Blur Detection",
            "Lighting Condition Analysis",
            "Real-time Quality Metrics"
        ],
        "Debugging & Visualization": [
            "Real-time Before/After Visualization",
            "Debug Frame Saving",
            "Processing Statistics",
            "Performance Monitoring"
        ]
    }
    
    for category, feature_list in features.items():
        print(f"\n{category}:")
        for feature in feature_list:
            print(f"  • {feature}")

if __name__ == "__main__":
    demonstrate_preprocessing()
    show_preprocessing_features()
    show_usage_examples()
    
    print("\n" + "=" * 50)
    print("For complete documentation, see the README.md file")
    print("For help with all options: python main.py --help")
    print("=" * 50)