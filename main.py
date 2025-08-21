import argparse
from body_measurement import BodyMeasurementSystem
from video_preprocessor import PreprocessingConfig

def create_preprocessing_config(args) -> PreprocessingConfig:
    """Create preprocessing configuration from command line arguments."""
    config = PreprocessingConfig()
    
    # Frame Enhancement
    config.enable_brightness_contrast = args.enable_brightness_contrast
    config.brightness_factor = args.brightness_factor
    config.contrast_factor = args.contrast_factor
    
    config.enable_noise_reduction = args.enable_noise_reduction
    config.noise_reduction_method = args.noise_reduction_method
    
    config.enable_sharpening = args.enable_sharpening
    config.sharpening_strength = args.sharpening_strength
    
    config.enable_histogram_equalization = args.enable_histogram_equalization
    config.hist_eq_method = args.hist_eq_method
    
    # Background Processing
    config.enable_background_subtraction = args.enable_background_subtraction
    config.background_method = args.background_method
    
    config.enable_edge_enhancement = args.enable_edge_enhancement
    config.edge_method = args.edge_method
    
    config.enable_adaptive_background = args.enable_adaptive_background
    
    # Color Space Optimization
    config.enable_color_space_conversion = args.enable_color_space_conversion
    config.target_color_space = args.target_color_space
    
    config.enable_skin_tone_enhancement = args.enable_skin_tone_enhancement
    
    # Stabilization
    config.enable_video_stabilization = args.enable_video_stabilization
    config.enable_frame_alignment = args.enable_frame_alignment
    
    # Quality Assessment
    config.enable_quality_assessment = args.enable_quality_assessment
    config.enable_motion_blur_detection = args.enable_motion_blur_detection
    config.enable_lighting_analysis = args.enable_lighting_analysis
    
    # Debugging and Visualization
    config.enable_visualization = args.enable_visualization
    config.save_debug_frames = args.save_debug_frames
    config.debug_output_dir = args.debug_output_dir
    
    return config

def main():
    parser = argparse.ArgumentParser(
        description='Calculate body measurements from a 360-degree video with optional preprocessing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Preprocessing Examples:
  Basic usage:
    python main.py --video video.mp4 --radius 150 --height 175
    
  Enable all preprocessing:
    python main.py --video video.mp4 --radius 150 --height 175 --enable-all-preprocessing
    
  Custom preprocessing:
    python main.py --video video.mp4 --radius 150 --height 175 \\
      --enable-brightness-contrast --brightness-factor 1.3 \\
      --enable-noise-reduction --noise-reduction-method bilateral \\
      --enable-background-subtraction --background-method mog2 \\
      --enable-skin-tone-enhancement
    
  Debug mode with visualization:
    python main.py --video video.mp4 --radius 150 --height 175 \\
      --enable-all-preprocessing --enable-visualization --save-debug-frames
        """
    )
    
    # Basic arguments
    parser.add_argument('--video', required=True, help='Path to the 360-degree video file')
    parser.add_argument('--radius', type=float, required=True, help='Camera rotation radius in cm')
    parser.add_argument('--height', type=float, help='Known height of the person in cm (for calibration)')
    parser.add_argument('--output', help='Path to save the 3D model visualization')
    
    # Preprocessing control
    parser.add_argument('--disable-preprocessing', action='store_true', 
                       help='Disable all preprocessing (use raw video frames)')
    parser.add_argument('--enable-all-preprocessing', action='store_true',
                       help='Enable all preprocessing features with default settings')
    
    # Frame Enhancement
    enhancement_group = parser.add_argument_group('Frame Enhancement')
    enhancement_group.add_argument('--enable-brightness-contrast', action='store_true',
                                 help='Enable brightness and contrast adjustment')
    enhancement_group.add_argument('--brightness-factor', type=float, default=1.2,
                                 help='Brightness adjustment factor (default: 1.2)')
    enhancement_group.add_argument('--contrast-factor', type=float, default=1.1,
                                 help='Contrast adjustment factor (default: 1.1)')
    
    enhancement_group.add_argument('--enable-noise-reduction', action='store_true',
                                 help='Enable noise reduction')
    enhancement_group.add_argument('--noise-reduction-method', choices=['bilateral', 'gaussian', 'median'],
                                 default='bilateral', help='Noise reduction method (default: bilateral)')
    
    enhancement_group.add_argument('--enable-sharpening', action='store_true',
                                 help='Enable sharpening filters')
    enhancement_group.add_argument('--sharpening-strength', type=float, default=0.5,
                                 help='Sharpening strength (0.0-1.0, default: 0.5)')
    
    enhancement_group.add_argument('--enable-histogram-equalization', action='store_true',
                                 help='Enable histogram equalization')
    enhancement_group.add_argument('--hist-eq-method', choices=['adaptive', 'standard'],
                                 default='adaptive', help='Histogram equalization method (default: adaptive)')
    
    # Background Processing
    background_group = parser.add_argument_group('Background Processing')
    background_group.add_argument('--enable-background-subtraction', action='store_true',
                                help='Enable background subtraction')
    background_group.add_argument('--background-method', choices=['mog2', 'knn'],
                                default='mog2', help='Background subtraction method (default: mog2)')
    
    background_group.add_argument('--enable-edge-enhancement', action='store_true',
                                help='Enable edge enhancement')
    background_group.add_argument('--edge-method', choices=['canny', 'sobel', 'laplacian'],
                                default='canny', help='Edge detection method (default: canny)')
    
    background_group.add_argument('--enable-adaptive-background', action='store_true',
                                help='Enable adaptive background modeling')
    
    # Color Space Optimization
    color_group = parser.add_argument_group('Color Space Optimization')
    color_group.add_argument('--enable-color-space-conversion', action='store_true',
                           help='Enable color space optimization')
    color_group.add_argument('--target-color-space', choices=['hsv', 'lab', 'rgb'],
                           default='hsv', help='Target color space (default: hsv)')
    
    color_group.add_argument('--enable-skin-tone-enhancement', action='store_true',
                           help='Enable skin tone enhancement')
    
    # Stabilization
    stabilization_group = parser.add_argument_group('Video Stabilization')
    stabilization_group.add_argument('--enable-video-stabilization', action='store_true',
                                   help='Enable video stabilization (computationally expensive)')
    stabilization_group.add_argument('--enable-frame-alignment', action='store_true',
                                   help='Enable frame alignment for consistent pose detection')
    
    # Quality Assessment
    quality_group = parser.add_argument_group('Quality Assessment')
    quality_group.add_argument('--enable-quality-assessment', action='store_true',
                             help='Enable frame quality assessment')
    quality_group.add_argument('--enable-motion-blur-detection', action='store_true',
                             help='Enable motion blur detection')
    quality_group.add_argument('--enable-lighting-analysis', action='store_true',
                             help='Enable lighting condition analysis')
    
    # Debugging and Visualization
    debug_group = parser.add_argument_group('Debugging and Visualization')
    debug_group.add_argument('--enable-visualization', action='store_true',
                           help='Enable real-time preprocessing visualization')
    debug_group.add_argument('--save-debug-frames', action='store_true',
                           help='Save before/after frames for debugging')
    debug_group.add_argument('--debug-output-dir', default='/tmp/preprocessing_debug',
                           help='Directory to save debug frames (default: /tmp/preprocessing_debug)')
    
    args = parser.parse_args()
    
    # Handle preprocessing configuration
    preprocessing_config = None
    
    if not args.disable_preprocessing:
        if args.enable_all_preprocessing:
            # Enable all preprocessing features with default settings
            preprocessing_config = PreprocessingConfig()
            preprocessing_config.enable_brightness_contrast = True
            preprocessing_config.enable_noise_reduction = True
            preprocessing_config.enable_sharpening = True
            preprocessing_config.enable_histogram_equalization = True
            preprocessing_config.enable_background_subtraction = True
            preprocessing_config.enable_edge_enhancement = True
            preprocessing_config.enable_adaptive_background = True
            preprocessing_config.enable_color_space_conversion = True
            preprocessing_config.enable_skin_tone_enhancement = True
            preprocessing_config.enable_frame_alignment = True
            preprocessing_config.enable_quality_assessment = True
            preprocessing_config.enable_motion_blur_detection = True
            preprocessing_config.enable_lighting_analysis = True
            preprocessing_config.enable_visualization = args.enable_visualization
            preprocessing_config.save_debug_frames = args.save_debug_frames
            preprocessing_config.debug_output_dir = args.debug_output_dir
        else:
            # Check if any preprocessing features are explicitly enabled
            preprocessing_enabled = any([
                args.enable_brightness_contrast,
                args.enable_noise_reduction,
                args.enable_sharpening,
                args.enable_histogram_equalization,
                args.enable_background_subtraction,
                args.enable_edge_enhancement,
                args.enable_adaptive_background,
                args.enable_color_space_conversion,
                args.enable_skin_tone_enhancement,
                args.enable_video_stabilization,
                args.enable_frame_alignment,
                args.enable_quality_assessment,
                args.enable_motion_blur_detection,
                args.enable_lighting_analysis,
                args.enable_visualization,
                args.save_debug_frames
            ])
            
            if preprocessing_enabled:
                preprocessing_config = create_preprocessing_config(args)
    
    # Initialize the system
    measurement_system = BodyMeasurementSystem(
        camera_radius=args.radius,
        calibration_height=args.height,
        preprocessing_config=preprocessing_config
    )
    
    # Process the video
    print(f"Processing video: {args.video}")
    measurement_system.process_video(args.video)
    
    # Calculate measurements
    measurements = measurement_system.calculate_measurements()
    
    # Print measurements
    print("\nBody Measurements:")
    print("-----------------")
    for measurement, value in measurements.items():
        print(f"{measurement.replace('_', ' ').title()}: {value:.2f} cm")
    
    # Visualize if requested
    if args.output:
        measurement_system.visualize_3d_model(args.output)
    else:
        measurement_system.visualize_3d_model()

if __name__ == "__main__":
    main()