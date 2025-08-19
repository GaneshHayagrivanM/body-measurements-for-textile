import argparse
from body_measurement import BodyMeasurementSystem

def main():
    parser = argparse.ArgumentParser(description='Calculate body measurements from a 360-degree video.')
    parser.add_argument('--video', required=True, help='Path to the 360-degree video file')
    parser.add_argument('--radius', type=float, required=True, help='Camera rotation radius in cm')
    parser.add_argument('--height', type=float, help='Known height of the person in cm (for calibration)')
    parser.add_argument('--output', help='Path to save the 3D model visualization')
    
    args = parser.parse_args()
    
    # Initialize the system
    measurement_system = BodyMeasurementSystem(
        camera_radius=args.radius,
        calibration_height=args.height
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