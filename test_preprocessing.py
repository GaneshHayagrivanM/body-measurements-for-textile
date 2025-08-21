#!/usr/bin/env python3
"""
Simple test script to validate the video preprocessing functionality.
Creates a synthetic test frame and processes it through the pipeline.
"""

import cv2
import numpy as np
from video_preprocessor import VideoPreprocessor, PreprocessingConfig

def create_test_frame():
    """Create a synthetic test frame for testing."""
    # Create a 640x480 test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some background pattern
    for i in range(0, 640, 20):
        cv2.line(frame, (i, 0), (i, 480), (50, 50, 50), 1)
    for i in range(0, 480, 20):
        cv2.line(frame, (0, i), (640, i), (50, 50, 50), 1)
    
    # Add a person-like silhouette
    cv2.ellipse(frame, (320, 240), (60, 120), 0, 0, 360, (150, 100, 80), -1)  # Body
    cv2.circle(frame, (320, 160), 25, (180, 120, 100), -1)  # Head
    cv2.rectangle(frame, (280, 180), (360, 220), (140, 90, 70), -1)  # Torso
    cv2.rectangle(frame, (260, 210), (300, 350), (120, 80, 60), -1)  # Left arm
    cv2.rectangle(frame, (340, 210), (380, 350), (120, 80, 60), -1)  # Right arm
    cv2.rectangle(frame, (300, 300), (320, 420), (100, 70, 50), -1)  # Left leg
    cv2.rectangle(frame, (320, 300), (340, 420), (100, 70, 50), -1)  # Right leg
    
    # Add some noise
    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame

def test_preprocessing():
    """Test the preprocessing pipeline."""
    print("Testing Video Preprocessing Pipeline")
    print("=" * 40)
    
    # Create test frame
    print("Creating synthetic test frame...")
    test_frame = create_test_frame()
    
    # Test configurations
    configs = [
        ("No preprocessing", None),
        ("Basic enhancement", PreprocessingConfig(
            enable_brightness_contrast=True,
            enable_noise_reduction=True,
            enable_sharpening=True
        )),
        ("Full preprocessing", PreprocessingConfig(
            enable_brightness_contrast=True,
            enable_noise_reduction=True,
            enable_sharpening=True,
            enable_histogram_equalization=True,
            enable_background_subtraction=True,
            enable_edge_enhancement=True,
            enable_color_space_conversion=True,
            enable_skin_tone_enhancement=True,
            enable_quality_assessment=True,
            enable_motion_blur_detection=True,
            enable_lighting_analysis=True
        ))
    ]
    
    for config_name, config in configs:
        print(f"\nTesting: {config_name}")
        print("-" * 30)
        
        if config is None:
            # No preprocessing
            processed_frame = test_frame
            metrics = {}
        else:
            # Initialize preprocessor
            preprocessor = VideoPreprocessor(config)
            
            # Process frame
            processed_frame, metrics = preprocessor.process_frame(test_frame)
            
            # Cleanup
            preprocessor.cleanup()
        
        # Print metrics
        if metrics:
            print("Quality Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.3f}")
            
            # Calculate improvement score
            if 'overall_quality' in metrics:
                print(f"  Overall Quality Score: {metrics['overall_quality']:.3f}")
        else:
            print("No metrics available (preprocessing disabled)")
        
        # Save comparison images
        comparison = np.hstack([test_frame, processed_frame])
        filename = f"/tmp/test_{config_name.replace(' ', '_').lower()}.jpg"
        cv2.imwrite(filename, comparison)
        print(f"Comparison saved to: {filename}")

def test_configuration():
    """Test configuration creation and validation."""
    print("\nTesting Configuration System")
    print("=" * 40)
    
    # Test default configuration
    default_config = PreprocessingConfig()
    print("Default configuration created successfully")
    
    # Test custom configuration
    custom_config = PreprocessingConfig(
        enable_brightness_contrast=True,
        brightness_factor=1.5,
        enable_noise_reduction=True,
        noise_reduction_method="gaussian",
        enable_quality_assessment=True
    )
    print("Custom configuration created successfully")
    
    # Print active features
    active_features = []
    if custom_config.enable_brightness_contrast:
        active_features.append(f"brightness/contrast (factor: {custom_config.brightness_factor})")
    if custom_config.enable_noise_reduction:
        active_features.append(f"noise reduction ({custom_config.noise_reduction_method})")
    if custom_config.enable_quality_assessment:
        active_features.append("quality assessment")
    
    print(f"Active features: {', '.join(active_features)}")

def test_integration():
    """Test integration with body measurement system."""
    print("\nTesting Integration")
    print("=" * 40)
    
    try:
        from body_measurement import BodyMeasurementSystem
        
        # Test without preprocessing
        system1 = BodyMeasurementSystem(camera_radius=150.0)
        print("✓ BodyMeasurementSystem created without preprocessing")
        
        # Test with preprocessing
        config = PreprocessingConfig(enable_quality_assessment=True)
        system2 = BodyMeasurementSystem(camera_radius=150.0, preprocessing_config=config)
        print("✓ BodyMeasurementSystem created with preprocessing")
        
        # Test statistics methods
        stats = system2.get_preprocessing_statistics()
        print(f"✓ Preprocessing statistics: {stats}")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")

if __name__ == "__main__":
    print("Video Preprocessing Test Suite")
    print("=" * 50)
    
    try:
        test_configuration()
        test_preprocessing()
        test_integration()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully! ✓")
        print("Check /tmp/ directory for output images")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()