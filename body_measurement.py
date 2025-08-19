import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import math
import os

class BodyMeasurementSystem:
    def __init__(self, camera_radius: float, calibration_height: float = None):
        """
        Initialize the body measurement system.
        
        Args:
            camera_radius: The constant radius of camera rotation (in cm)
            calibration_height: Optional known height of the person (in cm) for scaling
        """
        self.camera_radius = camera_radius
        self.calibration_height = calibration_height
        
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # For drawing the pose landmarks
        self.mp_drawing = mp.solutions.drawing_utils
        
        # For storing 3D points from all frames
        self.body_points_3d = []
        self.frame_angles = []
        self.landmark_tracks = {i: [] for i in range(33)}  # MediaPipe has 33 landmarks
        
        # Scale factor (to be calculated during processing)
        self.scale_factor = 1.0
        
    def process_video(self, video_path: str) -> None:
        """Process the 360-degree video to extract body points."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Assuming a full 360° rotation, calculate angle per frame
        angle_per_frame = 360 / total_frames
        
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Calculate the current angle based on frame index
            current_angle = frame_idx * angle_per_frame
            self.frame_angles.append(math.radians(current_angle))
            
            # Process the frame
            self._process_frame(frame, current_angle)
            
            frame_idx += 1
            
            # Display progress
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx}/{total_frames}")
                
        cap.release()
        
        # Reconstruct 3D model and calculate measurements
        self._reconstruct_3d_model()
        
    def _process_frame(self, frame: np.ndarray, angle: float) -> None:
        """Process a single frame to extract body landmarks."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            
            h, w, _ = frame.shape
            
            # Store landmarks in 3D space based on camera position
            for idx, landmark in enumerate(landmarks):
                # Convert to pixel coordinates
                x, y, z = landmark.x * w, landmark.y * h, landmark.z * w
                
                # Store in the landmark tracks
                self.landmark_tracks[idx].append((x, y, z, angle))
                
    def _reconstruct_3d_model(self) -> None:
        """Reconstruct 3D model from the tracked landmarks."""
        # Convert landmark tracks to 3D points in a cylindrical coordinate system
        self.model_3d = {}
        
        for landmark_idx, track in self.landmark_tracks.items():
            if not track:
                continue
                
            # Average the positions from different views for better accuracy
            positions = np.array([(pos[0], pos[1], pos[2], pos[3]) for pos in track])
            
            # Convert to 3D coordinates
            x_coords = []
            y_coords = []
            z_coords = []
            
            for pos in positions:
                x, y, z, angle = pos
                
                # Use camera radius and angle to determine 3D position
                # The exact transformation depends on camera setup and coordinate system
                x_3d = self.camera_radius * math.cos(angle) + x * math.cos(angle)
                y_3d = y  # Height (Y) remains the same
                z_3d = self.camera_radius * math.sin(angle) + x * math.sin(angle)
                
                x_coords.append(x_3d)
                y_coords.append(y_3d)
                z_coords.append(z_3d)
            
            # Average the 3D positions to get a single point
            avg_x = np.mean(x_coords)
            avg_y = np.mean(y_coords)
            avg_z = np.mean(z_coords)
            
            self.model_3d[landmark_idx] = (avg_x, avg_y, avg_z)
            
        # Calculate scale factor if calibration height is provided
        if self.calibration_height:
            # MediaPipe landmarks: 0=nose, 29=left heel, 30=right heel
            if 0 in self.model_3d and 29 in self.model_3d and 30 in self.model_3d:
                nose = self.model_3d[0]
                left_heel = self.model_3d[29]
                right_heel = self.model_3d[30]
                
                # Average the heel positions for the bottom point
                bottom_y = (left_heel[1] + right_heel[1]) / 2
                height_pixels = nose[1] - bottom_y
                
                # Calculate scale factor (cm per pixel)
                self.scale_factor = abs(self.calibration_height / height_pixels)
            
    def calculate_measurements(self) -> Dict[str, float]:
        """Calculate body measurements from the 3D model."""
        if not hasattr(self, 'model_3d') or not self.model_3d:
            raise ValueError("3D model has not been reconstructed yet. Run process_video first.")
            
        measurements = {}
        
        # Calculate height
        # MediaPipe landmarks: 0=nose, 29=left heel, 30=right heel
        if 0 in self.model_3d and 29 in self.model_3d and 30 in self.model_3d:
            nose = self.model_3d[0]
            left_heel = self.model_3d[29]
            right_heel = self.model_3d[30]
            
            # Use the lowest point of the heels
            bottom_y = max(left_heel[1], right_heel[1])  # Y increases downward in image coordinates
            height_model_units = abs(nose[1] - bottom_y)
            measurements['height'] = height_model_units * self.scale_factor  # in cm
        
        # Calculate shoulder width
        # MediaPipe landmarks: 11=left shoulder, 12=right shoulder
        if 11 in self.model_3d and 12 in self.model_3d:
            left_shoulder = self.model_3d[11]
            right_shoulder = self.model_3d[12]
            
            shoulder_width = math.sqrt(
                (left_shoulder[0] - right_shoulder[0])**2 + 
                (left_shoulder[2] - right_shoulder[2])**2
            )
            measurements['shoulders'] = shoulder_width * self.scale_factor  # in cm
        
        # Calculate waist
        # MediaPipe landmarks: 23=left hip, 24=right hip
        if 23 in self.model_3d and 24 in self.model_3d:
            left_hip = self.model_3d[23]
            right_hip = self.model_3d[24]
            
            # Estimate waist position (slightly above hips)
            waist_y = (left_hip[1] + right_hip[1]) / 2 - 10  # Adjust as needed
            
            # Find points around this height for waist circumference
            waist_points = []
            for angle in np.linspace(0, 2*np.pi, 36):  # Sample 36 points around
                # Estimate radius at each angle using body points
                x = np.mean([p[0] for p in self.model_3d.values() if abs(p[1] - waist_y) < 15])
                z = np.mean([p[2] for p in self.model_3d.values() if abs(p[1] - waist_y) < 15])
                
                # Adjust for body shape
                radius = math.sqrt(x**2 + z**2)
                
                waist_points.append((radius * math.cos(angle), waist_y, radius * math.sin(angle)))
            
            # Calculate waist circumference
            waist_circumference = self._calculate_circumference(waist_points)
            measurements['waist'] = waist_circumference * self.scale_factor  # in cm
        
        # Calculate hips
        # MediaPipe landmarks: 23=left hip, 24=right hip
        if 23 in self.model_3d and 24 in self.model_3d:
            left_hip = self.model_3d[23]
            right_hip = self.model_3d[24]
            
            # Hip position (below waist)
            hip_y = (left_hip[1] + right_hip[1]) / 2 + 15  # Adjust as needed
            
            # Find points around this height for hip circumference
            hip_points = []
            for angle in np.linspace(0, 2*np.pi, 36):  # Sample 36 points around
                # Estimate radius at each angle
                x = np.mean([p[0] for p in self.model_3d.values() if abs(p[1] - hip_y) < 15])
                z = np.mean([p[2] for p in self.model_3d.values() if abs(p[1] - hip_y) < 15])
                
                # Adjust for body shape
                radius = math.sqrt(x**2 + z**2)
                
                hip_points.append((radius * math.cos(angle), hip_y, radius * math.sin(angle)))
            
            # Calculate hip circumference
            hip_circumference = self._calculate_circumference(hip_points)
            measurements['hips'] = hip_circumference * self.scale_factor  # in cm
        
        # Additional measurements for clothing
        
        # Chest/bust
        # Estimate chest position (above waist, below shoulders)
        if 11 in self.model_3d and 12 in self.model_3d:
            left_shoulder = self.model_3d[11]
            right_shoulder = self.model_3d[12]
            
            chest_y = (left_shoulder[1] + right_shoulder[1]) / 2 + 25  # Adjust as needed
            
            chest_points = []
            for angle in np.linspace(0, 2*np.pi, 36):
                x = np.mean([p[0] for p in self.model_3d.values() if abs(p[1] - chest_y) < 15])
                z = np.mean([p[2] for p in self.model_3d.values() if abs(p[1] - chest_y) < 15])
                
                radius = math.sqrt(x**2 + z**2)
                chest_points.append((radius * math.cos(angle), chest_y, radius * math.sin(angle)))
            
            chest_circumference = self._calculate_circumference(chest_points)
            measurements['chest'] = chest_circumference * self.scale_factor  # in cm
        
        # Arm length
        # MediaPipe landmarks: 11=left shoulder, 13=left elbow, 15=left wrist
        if 11 in self.model_3d and 13 in self.model_3d and 15 in self.model_3d:
            left_shoulder = self.model_3d[11]
            left_elbow = self.model_3d[13]
            left_wrist = self.model_3d[15]
            
            upper_arm = math.sqrt(
                (left_shoulder[0] - left_elbow[0])**2 + 
                (left_shoulder[1] - left_elbow[1])**2 + 
                (left_shoulder[2] - left_elbow[2])**2
            )
            
            forearm = math.sqrt(
                (left_elbow[0] - left_wrist[0])**2 + 
                (left_elbow[1] - left_wrist[1])**2 + 
                (left_elbow[2] - left_wrist[2])**2
            )
            
            measurements['arm_length'] = (upper_arm + forearm) * self.scale_factor  # in cm
        
        # Inseam (inner leg length)
        # MediaPipe landmarks: 23=left hip, 25=left knee, 27=left ankle
        if 23 in self.model_3d and 25 in self.model_3d and 27 in self.model_3d:
            left_hip = self.model_3d[23]
            left_knee = self.model_3d[25]
            left_ankle = self.model_3d[27]
            
            thigh = math.sqrt(
                (left_hip[0] - left_knee[0])**2 + 
                (left_hip[1] - left_knee[1])**2 + 
                (left_hip[2] - left_knee[2])**2
            )
            
            calf = math.sqrt(
                (left_knee[0] - left_ankle[0])**2 + 
                (left_knee[1] - left_ankle[1])**2 + 
                (left_knee[2] - left_ankle[2])**2
            )
            
            measurements['inseam'] = (thigh + calf) * self.scale_factor  # in cm
        
        return measurements
        
    def _calculate_circumference(self, points: List[Tuple[float, float, float]]) -> float:
        """Calculate the circumference of a set of 3D points."""
        if len(points) < 3:
            return 0
            
        # Project points onto the XZ plane for waist/hip calculation
        points_2d = [(p[0], p[2]) for p in points]
        
        # Calculate convex hull
        hull = ConvexHull(points_2d)
        
        # Calculate perimeter of the hull
        perimeter = 0
        for simplex in hull.simplices:
            p1 = points_2d[simplex[0]]
            p2 = points_2d[simplex[1]]
            perimeter += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
        return perimeter
        
    def visualize_3d_model(self, save_path: str = None) -> None:
        """Visualize the 3D model with measurements."""
        if not hasattr(self, 'model_3d') or not self.model_3d:
            raise ValueError("3D model has not been reconstructed yet. Run process_video first.")
            
        # Create a point cloud from the model
        points = np.array(list(self.model_3d.values()))
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors based on body parts
        colors = np.zeros((len(points), 3))
        colors[:, 0] = 0.8  # Default red tint
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd], 
                                          window_name="3D Body Model",
                                          width=800, height=600)
        
        if save_path:
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"3D model saved to {save_path}")