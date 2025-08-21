"""
Video Preprocessing Pipeline for Body Measurement System

This module provides comprehensive video preprocessing capabilities to enhance
body detection accuracy in the textile measurement system.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass
from skimage import exposure, restoration, measure, filters
from skimage.util import random_noise
from skimage.color import rgb2hsv, rgb2lab, hsv2rgb, lab2rgb
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration class for video preprocessing options."""
    
    # Frame Enhancement
    enable_brightness_contrast: bool = True
    brightness_factor: float = 1.2
    contrast_factor: float = 1.1
    
    enable_noise_reduction: bool = True
    noise_reduction_method: str = "bilateral"  # "bilateral", "gaussian", "median"
    
    enable_sharpening: bool = True
    sharpening_strength: float = 0.5
    
    enable_histogram_equalization: bool = True
    hist_eq_method: str = "adaptive"  # "adaptive", "standard"
    
    # Background Processing
    enable_background_subtraction: bool = True
    background_method: str = "mog2"  # "mog2", "knn", "gmg"
    
    enable_edge_enhancement: bool = True
    edge_method: str = "canny"  # "canny", "sobel", "laplacian"
    
    enable_adaptive_background: bool = True
    
    # Color Space Optimization
    enable_color_space_conversion: bool = True
    target_color_space: str = "hsv"  # "hsv", "lab", "rgb"
    
    enable_skin_tone_enhancement: bool = True
    
    # Stabilization
    enable_video_stabilization: bool = False  # Computationally expensive
    enable_frame_alignment: bool = True
    
    # Quality Assessment
    enable_quality_assessment: bool = True
    enable_motion_blur_detection: bool = True
    enable_lighting_analysis: bool = True
    
    # Debugging and Visualization
    enable_visualization: bool = False
    save_debug_frames: bool = False
    debug_output_dir: str = "/tmp/preprocessing_debug"


class VideoPreprocessor:
    """
    Comprehensive video preprocessing pipeline for enhanced body detection.
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        """
        Initialize the video preprocessor.
        
        Args:
            config: Preprocessing configuration object
        """
        self.config = config if config else PreprocessingConfig()
        self.background_subtractor = None
        self.previous_frame = None
        self.frame_count = 0
        self.quality_scores = []
        
        # Initialize background subtractor if enabled
        if self.config.enable_background_subtraction:
            self._initialize_background_subtractor()
        
        # Create debug directory if needed
        if self.config.save_debug_frames:
            import os
            os.makedirs(self.config.debug_output_dir, exist_ok=True)
    
    def _initialize_background_subtractor(self):
        """Initialize background subtraction algorithm."""
        if self.config.background_method == "mog2":
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                detectShadows=True, varThreshold=16
            )
        elif self.config.background_method == "knn":
            self.background_subtractor = cv2.createBackgroundSubtractorKNN(
                detectShadows=True
            )
        else:
            logger.warning(f"Unknown background method: {self.config.background_method}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame through the preprocessing pipeline.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, quality_metrics)
        """
        start_time = time.time()
        original_frame = frame.copy()
        processed_frame = frame.copy()
        quality_metrics = {}
        
        try:
            # 1. Frame Enhancement
            if self.config.enable_brightness_contrast:
                processed_frame = self._adjust_brightness_contrast(processed_frame)
            
            if self.config.enable_noise_reduction:
                processed_frame = self._reduce_noise(processed_frame)
            
            if self.config.enable_sharpening:
                processed_frame = self._apply_sharpening(processed_frame)
            
            if self.config.enable_histogram_equalization:
                processed_frame = self._apply_histogram_equalization(processed_frame)
            
            # 2. Background Processing
            if self.config.enable_background_subtraction:
                processed_frame = self._apply_background_subtraction(processed_frame)
            
            if self.config.enable_edge_enhancement:
                processed_frame = self._enhance_edges(processed_frame)
            
            # 3. Color Space Optimization
            if self.config.enable_color_space_conversion:
                processed_frame = self._optimize_color_space(processed_frame)
            
            if self.config.enable_skin_tone_enhancement:
                processed_frame = self._enhance_skin_tone(processed_frame)
            
            # 4. Stabilization (if enabled)
            if self.config.enable_frame_alignment and self.previous_frame is not None:
                processed_frame = self._align_frame(processed_frame)
            
            # 5. Quality Assessment
            if self.config.enable_quality_assessment:
                quality_metrics = self._assess_frame_quality(processed_frame)
            
            # Update tracking variables
            self.previous_frame = processed_frame.copy()
            self.frame_count += 1
            
            # Add processing time to metrics
            quality_metrics['processing_time'] = time.time() - start_time
            
            # Save debug frames if enabled
            if self.config.save_debug_frames:
                self._save_debug_frames(original_frame, processed_frame)
            
            # Visualization if enabled
            if self.config.enable_visualization:
                self._visualize_processing(original_frame, processed_frame, quality_metrics)
            
            return processed_frame, quality_metrics
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_count}: {str(e)}")
            return original_frame, {'error': str(e)}
    
    def _adjust_brightness_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast of the frame."""
        # Convert to float for precise calculations
        frame_float = frame.astype(np.float32)
        
        # Apply brightness and contrast adjustments
        adjusted = frame_float * self.config.contrast_factor + \
                  (self.config.brightness_factor - 1.0) * 50
        
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    def _reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction to the frame."""
        if self.config.noise_reduction_method == "bilateral":
            return cv2.bilateralFilter(frame, 9, 75, 75)
        elif self.config.noise_reduction_method == "gaussian":
            return cv2.GaussianBlur(frame, (5, 5), 0)
        elif self.config.noise_reduction_method == "median":
            return cv2.medianBlur(frame, 5)
        else:
            return frame
    
    def _apply_sharpening(self, frame: np.ndarray) -> np.ndarray:
        """Apply sharpening filter to enhance edges."""
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        
        # Apply sharpening
        sharpened = cv2.filter2D(frame, -1, kernel)
        
        # Blend with original based on strength
        strength = self.config.sharpening_strength
        return cv2.addWeighted(frame, 1 - strength, sharpened, strength, 0)
    
    def _apply_histogram_equalization(self, frame: np.ndarray) -> np.ndarray:
        """Apply histogram equalization for better exposure."""
        if self.config.hist_eq_method == "adaptive":
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Standard histogram equalization on each channel
            channels = cv2.split(frame)
            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
            return cv2.merge(eq_channels)
    
    def _apply_background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """Apply background subtraction to isolate foreground."""
        if self.background_subtractor is None:
            return frame
        
        # Get foreground mask
        fg_mask = self.background_subtractor.apply(frame)
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to frame
        result = cv2.bitwise_and(frame, frame, mask=fg_mask)
        
        # Blend with original frame to preserve some background info
        return cv2.addWeighted(frame, 0.3, result, 0.7, 0)
    
    def _enhance_edges(self, frame: np.ndarray) -> np.ndarray:
        """Enhance edges around human silhouette."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.config.edge_method == "canny":
            edges = cv2.Canny(gray, 50, 150)
        elif self.config.edge_method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        else:
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        
        # Convert edges to 3-channel for blending
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Enhance edges in the original frame
        return cv2.addWeighted(frame, 0.8, edges_3ch, 0.2, 0)
    
    def _optimize_color_space(self, frame: np.ndarray) -> np.ndarray:
        """Convert to optimal color space for human detection."""
        if self.config.target_color_space == "hsv":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Enhance saturation for better skin detection
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif self.config.target_color_space == "lab":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            # Enhance A and B channels for better color separation
            lab[:, :, 1] = cv2.multiply(lab[:, :, 1], 1.1)
            lab[:, :, 2] = cv2.multiply(lab[:, :, 2], 1.1)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return frame
    
    def _enhance_skin_tone(self, frame: np.ndarray) -> np.ndarray:
        """Enhance skin tone regions for better human detection."""
        # Convert to HSV for skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin tone range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Enhance skin regions
        enhanced = frame.copy()
        enhanced[skin_mask > 0] = cv2.multiply(enhanced[skin_mask > 0], 1.15)
        
        return enhanced
    
    def _align_frame(self, frame: np.ndarray) -> np.ndarray:
        """Align frame with previous frame for consistent pose detection."""
        if self.previous_frame is None:
            return frame
        
        # Convert frames to grayscale for feature detection
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ORB features
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray_previous, None)
        kp2, des2 = orb.detectAndCompute(gray_current, None)
        
        if des1 is None or des2 is None:
            return frame
        
        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        
        if len(matches) < 10:
            return frame
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
        
        # Find homography
        try:
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = frame.shape[:2]
                aligned = cv2.warpPerspective(frame, M, (w, h))
                return aligned
        except:
            pass
        
        return frame
    
    def _assess_frame_quality(self, frame: np.ndarray) -> Dict[str, float]:
        """Assess the quality of the processed frame."""
        metrics = {}
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness (using Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'] = float(laplacian_var) if laplacian_var is not None else 0.0
            
            # 2. Brightness
            brightness = np.mean(gray)
            metrics['brightness'] = float(brightness) if brightness is not None else 128.0
            
            # 3. Contrast (standard deviation)
            contrast = np.std(gray)
            metrics['contrast'] = float(contrast) if contrast is not None else 0.0
            
            # 4. Motion blur detection (if enabled)
            if self.config.enable_motion_blur_detection:
                try:
                    metrics['motion_blur'] = self._detect_motion_blur(gray)
                except Exception as e:
                    logger.warning(f"Motion blur detection failed: {e}")
                    metrics['motion_blur'] = 0.0
            
            # 5. Lighting condition analysis (if enabled)
            if self.config.enable_lighting_analysis:
                try:
                    lighting_metrics = self._analyze_lighting(gray)
                    metrics.update(lighting_metrics)
                except Exception as e:
                    logger.warning(f"Lighting analysis failed: {e}")
                    metrics.update({'overexposure': 0.0, 'underexposure': 0.0, 'dynamic_range': 0.0})
            
            # 6. Overall quality score (weighted combination)
            sharpness = max(0.0, min(metrics.get('sharpness', 0.0) / 1000, 1.0))
            brightness_score = max(0.0, min(1.0 - abs(metrics.get('brightness', 128.0) - 128) / 128, 1.0))
            contrast_score = max(0.0, min(metrics.get('contrast', 0.0) / 50, 1.0))
            
            quality_score = 0.4 * sharpness + 0.3 * brightness_score + 0.3 * contrast_score
            metrics['overall_quality'] = quality_score
            
            self.quality_scores.append(quality_score)
            
        except Exception as e:
            logger.error(f"Frame quality assessment failed: {e}")
            metrics = {
                'sharpness': 0.0,
                'brightness': 128.0,
                'contrast': 0.0,
                'overall_quality': 0.0,
                'error': str(e)
            }
            self.quality_scores.append(0.0)
        
        return metrics
    
    def _detect_motion_blur(self, gray: np.ndarray) -> float:
        """Detect motion blur in the frame."""
        try:
            # Use FFT to detect motion blur
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Calculate the variance of the magnitude spectrum
            blur_score = np.var(magnitude_spectrum)
            return float(blur_score) if blur_score is not None else 0.0
        except Exception as e:
            logger.warning(f"Motion blur detection failed: {e}")
            return 0.0
    
    def _analyze_lighting(self, gray: np.ndarray) -> Dict[str, float]:
        """Analyze lighting conditions in the frame."""
        lighting_metrics = {}
        
        try:
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-8)  # Normalize with epsilon to avoid division by zero
            
            # Check for overexposure (bright pixels)
            lighting_metrics['overexposure'] = float(np.sum(hist[240:]))
            
            # Check for underexposure (dark pixels)
            lighting_metrics['underexposure'] = float(np.sum(hist[:16]))
            
            # Calculate dynamic range
            non_zero_indices = np.where(hist > 0.01)[0]
            if len(non_zero_indices) > 0:
                lighting_metrics['dynamic_range'] = float(non_zero_indices[-1] - non_zero_indices[0])
            else:
                lighting_metrics['dynamic_range'] = 0.0
                
        except Exception as e:
            logger.warning(f"Lighting analysis failed: {e}")
            lighting_metrics = {
                'overexposure': 0.0,
                'underexposure': 0.0,
                'dynamic_range': 0.0
            }
        
        return lighting_metrics
    
    def _save_debug_frames(self, original: np.ndarray, processed: np.ndarray):
        """Save original and processed frames for debugging."""
        import os
        
        # Create comparison image
        comparison = np.hstack([original, processed])
        
        # Save frame
        filename = f"frame_{self.frame_count:06d}_comparison.jpg"
        filepath = os.path.join(self.config.debug_output_dir, filename)
        cv2.imwrite(filepath, comparison)
    
    def _visualize_processing(self, original: np.ndarray, processed: np.ndarray, 
                            metrics: Dict[str, Any]):
        """Visualize the processing results in real-time."""
        # Create side-by-side comparison
        comparison = np.hstack([original, processed])
        
        # Add text overlay with metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                text = f"{key}: {value:.2f}"
                cv2.putText(comparison, text, (10, y_offset), font, 0.6, (0, 255, 0), 2)
                y_offset += 25
        
        # Display the result
        cv2.imshow('Original vs Processed', comparison)
        cv2.waitKey(1)
    
    def get_quality_statistics(self) -> Dict[str, float]:
        """Get statistics about frame quality throughout processing."""
        if not self.quality_scores:
            return {}
        
        return {
            'mean_quality': np.mean(self.quality_scores),
            'std_quality': np.std(self.quality_scores),
            'min_quality': np.min(self.quality_scores),
            'max_quality': np.max(self.quality_scores),
            'frames_processed': len(self.quality_scores)
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.config.enable_visualization:
            cv2.destroyAllWindows()