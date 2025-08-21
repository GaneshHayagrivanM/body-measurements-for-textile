import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import math
import os
from video_preprocessor import VideoPreprocessor, PreprocessingConfig

class BodyMeasurementSystem:
    def __init__(self, camera_radius: float, calibration_height: float = None, 
                 preprocessing_config: PreprocessingConfig = None):
        """
        Initialize the body measurement system.
        
        Args:
            camera_radius: The constant radius of camera rotation (in cm)
            calibration_height: Optional known height of the person (in cm) for scaling
            preprocessing_config: Optional preprocessing configuration
        """
        self.camera_radius = camera_radius
        self.calibration_height = calibration_height
        
        # Initialize preprocessing
        self.preprocessing_config = preprocessing_config
        self.preprocessor = None
        if preprocessing_config:
            self.preprocessor = VideoPreprocessor(preprocessing_config)
        
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
        
        # For storing partial point clouds from each frame
        self.partial_pcds = []
        
        # Scale factor (to be calculated during processing)
        self.scale_factor = 1.0
        # Final reconstructed model
        self.model = None
        
        # Store preprocessing metrics
        self.preprocessing_metrics = []
        
    def process_video(self, video_path: str) -> None:
        """Process the 360-degree video to extract body points."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError("Video file could not be opened or is empty.")

        # Assuming a full 360° rotation, calculate angle per frame
        angle_per_frame = 360 / total_frames
        
        # Print preprocessing status
        if self.preprocessor:
            print(f"Video preprocessing enabled with the following features:")
            config = self.preprocessing_config
            enabled_features = []
            if config.enable_brightness_contrast: enabled_features.append("brightness/contrast adjustment")
            if config.enable_noise_reduction: enabled_features.append("noise reduction")
            if config.enable_sharpening: enabled_features.append("sharpening")
            if config.enable_histogram_equalization: enabled_features.append("histogram equalization")
            if config.enable_background_subtraction: enabled_features.append("background subtraction")
            if config.enable_edge_enhancement: enabled_features.append("edge enhancement")
            if config.enable_color_space_conversion: enabled_features.append("color space optimization")
            if config.enable_skin_tone_enhancement: enabled_features.append("skin tone enhancement")
            if config.enable_frame_alignment: enabled_features.append("frame alignment")
            if config.enable_quality_assessment: enabled_features.append("quality assessment")
            
            for feature in enabled_features:
                print(f"  - {feature}")
        else:
            print("Video preprocessing disabled - using raw frames")
        
        frame_idx = 0
        preprocessing_times = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Calculate the current angle based on frame index (in radians)
            current_angle = math.radians(frame_idx * angle_per_frame)
            
            # Apply preprocessing if enabled
            processed_frame = frame
            if self.preprocessor:
                processed_frame, metrics = self.preprocessor.process_frame(frame)
                self.preprocessing_metrics.append(metrics)
                if 'processing_time' in metrics:
                    preprocessing_times.append(metrics['processing_time'])
            
            # Process the frame
            self._process_frame(processed_frame, current_angle)
            
            frame_idx += 1
            
            # Display progress
            if frame_idx % 10 == 0:
                progress_msg = f"Processing frame {frame_idx}/{total_frames}"
                if self.preprocessor and preprocessing_times:
                    avg_preprocess_time = np.mean(preprocessing_times[-10:])  # Last 10 frames
                    progress_msg += f" (avg preprocessing: {avg_preprocess_time:.3f}s)"
                print(progress_msg)
                
        cap.release()
        
        # Clean up preprocessing resources
        if self.preprocessor:
            self.preprocessor.cleanup()
            
        if not self.partial_pcds:
            print("Warning: No human pose detected in any frame. Cannot generate a model.")
            return

        # Reconstruct 3D model from the collected point clouds
        self._reconstruct_3d_model()
        
        # Print preprocessing summary if enabled
        if self.preprocessor:
            self.print_preprocessing_summary()
        
    def _process_frame(self, frame: np.ndarray, angle: float) -> None:
        """
        Process a single frame to extract a partial 3D point cloud.
        Uses MediaPipe's pose_world_landmarks for 3D coordinates.
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if results.pose_world_landmarks:
            # Get landmarks in their own 3D coordinate system
            landmarks = results.pose_world_landmarks.landmark
            points_3d = []
            
            for landmark in landmarks:
                # MediaPipe's world landmarks are in meters, with y-axis down.
                # We'll flip the y-axis to have y-up.
                points_3d.append([landmark.x, -landmark.y, landmark.z])
            
            points_3d = np.array(points_3d)
            
            # Create the rotation matrix to transform points to the world frame
            # The angle is the camera's position on the XZ plane, so we rotate around Y
            rotation_matrix = np.array([
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)]
            ])
            
            # Apply the rotation
            transformed_points = points_3d @ rotation_matrix.T
            
            # Create an Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(transformed_points)
            
            # Add the partial point cloud to our list
            self.partial_pcds.append(pcd)
                
    def _reconstruct_3d_model(self) -> None:
        """
        Reconstruct a full 3D model by aligning and merging partial point clouds.
        """
        print("Reconstructing 3D model...")
        
        # The first point cloud is the base
        global_pcd = self.partial_pcds[0]
        
        # ICP parameters
        threshold = 0.02  # 2cm distance threshold for correspondence
        trans_init = np.identity(4) # Initial transformation guess

        # Keep track of landmark positions through the transformations
        landmark_positions = {i: [] for i in range(33)}
        for i, p in enumerate(np.asarray(global_pcd.points)):
            landmark_positions[i].append(p)

        # Iteratively align each subsequent point cloud to the global one
        for i in range(1, len(self.partial_pcds)):
            source_pcd = self.partial_pcds[i]
            
            # Run ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pcd, global_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )
            
            # Transform the source point cloud with the calculated alignment
            source_pcd.transform(reg_p2p.transformation)
            
            # Record the transformed landmark positions
            for lm_idx, p in enumerate(np.asarray(source_pcd.points)):
                landmark_positions[lm_idx].append(p)

            # Merge with the global point cloud
            global_pcd += source_pcd
        
        # Calculate the final average position for each landmark
        self.final_landmarks = {
            i: np.mean(positions, axis=0)
            for i, positions in landmark_positions.items() if positions
        }

        # Downsample the point cloud to make it uniform
        # Voxel size of 5mm
        uniform_pcd = global_pcd.voxel_down_sample(voxel_size=0.005)

        # Remove statistical outliers to clean up noise
        pcd_cleaned, ind = uniform_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.model = pcd_cleaned.select_by_index(ind)

        print("Point cloud processed. Starting surface reconstruction...")

        # Estimate normals for the point cloud
        self.model.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
        # Perform surface reconstruction using Ball Pivoting Algorithm
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            self.model, o3d.utility.DoubleVector(radii))
            
        # The result is a mesh, which we store as our model
        self.model = mesh
        
        print("3D model reconstruction complete.")
        
        # Calculate scale factor if calibration height is provided
        if self.calibration_height:
            # The model points are in meters. Find the height of the model.
            points = np.asarray(self.model.vertices)
            min_y = np.min(points[:, 1])
            max_y = np.max(points[:, 1])
            
            model_height = max_y - min_y
            
            if model_height > 0:
                # The scale factor converts from model units (meters) to cm
                self.scale_factor = (self.calibration_height / model_height)
            else:
                print("Warning: Model height is zero, cannot calculate scale factor.")
                self.scale_factor = 100.0 # Fallback to convert meters to cm
        else:
            # If no calibration height, assume model is at real scale and convert meters to cm
            self.scale_factor = 100.0

    def calculate_measurements(self) -> Dict[str, float]:
        """Calculate body measurements from the reconstructed 3D model."""
        if self.model is None or not hasattr(self, 'final_landmarks'):
            raise ValueError("3D model has not been reconstructed yet. Run process_video first.")
            
        measurements = {}
        
        # Height: from the mesh's bounding box
        bounding_box = self.model.get_axis_aligned_bounding_box()
        model_height = bounding_box.get_extent()[1]
        measurements['height'] = model_height * self.scale_factor

        # Shoulder width
        if 11 in self.final_landmarks and 12 in self.final_landmarks:
            p1 = self.final_landmarks[11]
            p2 = self.final_landmarks[12]
            dist = np.linalg.norm(p1 - p2)
            measurements['shoulders'] = dist * self.scale_factor

        # Arm length (left arm)
        if 11 in self.final_landmarks and 13 in self.final_landmarks and 15 in self.final_landmarks:
            shoulder, elbow, wrist = self.final_landmarks[11], self.final_landmarks[13], self.final_landmarks[15]
            upper_arm = np.linalg.norm(shoulder - elbow)
            forearm = np.linalg.norm(elbow - wrist)
            measurements['arm_length'] = (upper_arm + forearm) * self.scale_factor
            
        # Inseam (left leg)
        if 23 in self.final_landmarks and 25 in self.final_landmarks and 27 in self.final_landmarks:
            hip, knee, ankle = self.final_landmarks[23], self.final_landmarks[25], self.final_landmarks[27]
            thigh = np.linalg.norm(hip - knee)
            calf = np.linalg.norm(knee - ankle)
            measurements['inseam'] = (thigh + calf) * self.scale_factor
            
        # Circumference measurements
        if 11 in self.final_landmarks and 12 in self.final_landmarks and \
           23 in self.final_landmarks and 24 in self.final_landmarks:
            
            # Estimate key body heights based on landmark positions
            shoulder_y = (self.final_landmarks[11][1] + self.final_landmarks[12][1]) / 2
            hip_y = (self.final_landmarks[23][1] + self.final_landmarks[24][1]) / 2
            
            # Chest is slightly below shoulders
            chest_y = shoulder_y - (shoulder_y - hip_y) * 0.2
            measurements['chest'] = self._calculate_circumference_at_y(chest_y) * self.scale_factor
            
            # Waist is roughly halfway between shoulders and hips
            waist_y = shoulder_y - (shoulder_y - hip_y) * 0.5
            measurements['waist'] = self._calculate_circumference_at_y(waist_y) * self.scale_factor
            
            # Hips
            measurements['hips'] = self._calculate_circumference_at_y(hip_y) * self.scale_factor

        return measurements
        
    def _calculate_circumference_at_y(self, y_level: float, slice_thickness: float = 0.02) -> float:
        """
        Calculates the circumference of the model at a specific y-level.
        It does this by taking a thin slice of the model's vertices,
        projecting them to the XZ plane, and finding the perimeter of their convex hull.
        """
        if self.model is None or not isinstance(self.model, o3d.geometry.TriangleMesh):
            return 0
            
        vertices = np.asarray(self.model.vertices)
        
        # Select points within the horizontal slice
        slice_indices = np.where(np.abs(vertices[:, 1] - y_level) < (slice_thickness / 2))[0]
        
        if len(slice_indices) < 3:
            return 0 # Not enough points to form a polygon
            
        slice_points = vertices[slice_indices]
        
        # Project points onto the XZ plane
        points_2d = slice_points[:, [0, 2]]
        
        try:
            # Calculate convex hull
            hull = ConvexHull(points_2d)
            # Calculate perimeter of the hull. For 2D hulls in scipy,
            # .perimeter is the correct attribute for the perimeter length.
            return hull.perimeter
        except Exception:
            # ConvexHull can fail if points are collinear
            return 0
        
    def visualize_3d_model(self, save_path: str = None) -> None:
        """Visualize the reconstructed 3D mesh."""
        if self.model is None or not isinstance(self.model, o3d.geometry.TriangleMesh):
            print("Model has not been reconstructed as a mesh, cannot visualize.")
            return

        # Give the mesh a nice color
        self.model.paint_uniform_color([0.7, 0.7, 0.7])
        # Compute normals for better rendering
        self.model.compute_vertex_normals()
        
        # Visualize the mesh
        o3d.visualization.draw_geometries([self.model],
                                          window_name="Reconstructed 3D Body Model",
                                          width=800, height=600)
        
        if save_path:
            o3d.io.write_triangle_mesh(save_path, self.model)
            print(f"3D mesh model saved to {save_path}")
    
    def get_preprocessing_statistics(self) -> Dict[str, any]:
        """Get comprehensive preprocessing statistics."""
        if not self.preprocessor or not self.preprocessing_metrics:
            return {"preprocessing_enabled": False}
        
        stats = {"preprocessing_enabled": True}
        
        # Get quality statistics from preprocessor
        quality_stats = self.preprocessor.get_quality_statistics()
        stats.update(quality_stats)
        
        # Calculate average metrics across all frames
        if self.preprocessing_metrics:
            # Collect all numeric metrics
            metric_keys = set()
            for metrics in self.preprocessing_metrics:
                metric_keys.update(k for k, v in metrics.items() if isinstance(v, (int, float)))
            
            for key in metric_keys:
                values = [m[key] for m in self.preprocessing_metrics if key in m and isinstance(m[key], (int, float))]
                if values:
                    stats[f"avg_{key}"] = np.mean(values)
                    stats[f"std_{key}"] = np.std(values)
                    stats[f"min_{key}"] = np.min(values)
                    stats[f"max_{key}"] = np.max(values)
        
        return stats
    
    def print_preprocessing_summary(self):
        """Print a summary of preprocessing performance and quality metrics."""
        stats = self.get_preprocessing_statistics()
        
        if not stats.get("preprocessing_enabled", False):
            print("Preprocessing was not enabled for this video.")
            return
        
        print("\n" + "="*50)
        print("VIDEO PREPROCESSING SUMMARY")
        print("="*50)
        
        # Quality metrics
        if 'mean_quality' in stats:
            print(f"Overall Quality Score: {stats['mean_quality']:.3f} ± {stats.get('std_quality', 0):.3f}")
            print(f"Quality Range: {stats.get('min_quality', 0):.3f} - {stats.get('max_quality', 0):.3f}")
        
        # Performance metrics
        if 'avg_processing_time' in stats:
            print(f"Average Processing Time per Frame: {stats['avg_processing_time']:.3f}s")
            print(f"Processing Time Range: {stats.get('min_processing_time', 0):.3f}s - {stats.get('max_processing_time', 0):.3f}s")
        
        # Image quality metrics
        if 'avg_sharpness' in stats:
            print(f"Average Sharpness: {stats['avg_sharpness']:.1f}")
        if 'avg_brightness' in stats:
            print(f"Average Brightness: {stats['avg_brightness']:.1f}")
        if 'avg_contrast' in stats:
            print(f"Average Contrast: {stats['avg_contrast']:.1f}")
        
        # Motion and lighting analysis
        if 'avg_motion_blur' in stats:
            print(f"Motion Blur Score: {stats['avg_motion_blur']:.3f}")
        if 'avg_overexposure' in stats:
            print(f"Overexposure: {stats['avg_overexposure']:.3f}")
        if 'avg_underexposure' in stats:
            print(f"Underexposure: {stats['avg_underexposure']:.3f}")
        
        print(f"Total Frames Processed: {stats.get('frames_processed', 0)}")
        print("="*50)
    
    def cleanup_preprocessing(self):
        """Clean up preprocessing resources."""
        if self.preprocessor:
            self.preprocessor.cleanup()