# Body Measurement System from 360° Video

## Program Overview

This solution uses computer vision and 3D reconstruction to extract accurate body measurements for clothing purposes:

1. Process the video to extract frames
2. Detect the human body and key points in each frame
3. Reconstruct a 3D model using the known camera radius
4. Calculate measurements: height, waist, shoulders, hips, etc.

## How to Use

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the program with your video:
   ```
   python main.py --video path_to_360_video.mp4 --radius 150 --height 175
   ```

   - `--video`: Path to your 360-degree video file
   - `--radius`: The camera rotation radius in centimeters
   - `--height`: Optional known height of the person for precise calibration
   - `--output`: Optional path to save the 3D model visualization

## Technical Details

- **Human Detection**: Uses MediaPipe for accurate human pose estimation
- **3D Reconstruction**: Combines multiple 2D views to create a complete 3D model
- **Measurement Calculation**: 
  - Height: Vertical distance from top of head to floor
  - Waist: Circumference at the narrowest part of the torso
  - Shoulders: Distance between shoulder joints
  - Hips: Circumference at the widest part of the lower body
  - Additional measurements for clothing purposes (chest, arm length, inseam)

## Limitations

- Requires good lighting and contrast between the person and background
- The person should remain relatively still during the video capture
- Loose or baggy clothing may reduce measurement accuracy
- Camera radius must be accurately known for proper scaling

## Future Improvements

- Implement more advanced 3D reconstruction techniques
- Add automatic calibration without requiring known measurements
- Support for multiple people in the video
- Generate a complete 3D model for virtual try-on applications