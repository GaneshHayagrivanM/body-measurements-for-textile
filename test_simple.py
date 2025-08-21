#!/usr/bin/env python3
"""
Simple test for video preprocessing without MediaPipe dependencies.
"""

import cv2
import numpy as np
from video_preprocessor import VideoPreprocessor, PreprocessingConfig

def create_test_frame():
    """Create a synthetic test frame."""
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add a simple shape
    cv2.rectangle(frame, (200, 150), (400, 350), (100, 150, 200), -1)
    cv2.circle(frame, (300, 100), 50, (150, 100, 50), -1)
    
    return frame

def test_basic_preprocessing():
    """Test basic preprocessing functionality."""
    print("Testing Basic Preprocessing")
    print("=" * 30)
    
    frame = create_test_frame()
    
    # Test minimal config
    config = PreprocessingConfig(
        enable_brightness_contrast=True,
        enable_noise_reduction=True,
        enable_quality_assessment=True
    )
    
    preprocessor = VideoPreprocessor(config)
    
    try:
        processed_frame, metrics = preprocessor.process_frame(frame)
        
        print("✓ Frame processed successfully")
        print(f"✓ Metrics generated: {len(metrics)} items")
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.3f}")
        
        preprocessor.cleanup()
        print("✓ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_all_features():
    """Test all preprocessing features individually."""
    print("\nTesting Individual Features")
    print("=" * 30)
    
    frame = create_test_frame()
    
    features = [
        ("brightness_contrast", {"enable_brightness_contrast": True}),
        ("noise_reduction", {"enable_noise_reduction": True}),
        ("sharpening", {"enable_sharpening": True}),
        ("histogram_equalization", {"enable_histogram_equalization": True}),
        ("edge_enhancement", {"enable_edge_enhancement": True}),
        ("color_space_conversion", {"enable_color_space_conversion": True}),
        ("skin_tone_enhancement", {"enable_skin_tone_enhancement": True}),
        ("quality_assessment", {"enable_quality_assessment": True})
    ]
    
    for feature_name, feature_config in features:
        try:
            config = PreprocessingConfig(**feature_config)
            preprocessor = VideoPreprocessor(config)
            
            processed_frame, metrics = preprocessor.process_frame(frame)
            preprocessor.cleanup()
            
            print(f"✓ {feature_name}")
            
        except Exception as e:
            print(f"✗ {feature_name}: {e}")

if __name__ == "__main__":
    print("Simple Preprocessing Test")
    print("=" * 50)
    
    success1 = test_basic_preprocessing()
    test_all_features()
    
    if success1:
        print("\n✓ All basic tests passed!")
    else:
        print("\n✗ Some tests failed!")