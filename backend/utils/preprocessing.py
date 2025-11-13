"""
Preprocessing Module
Handles image and video preprocessing for deepfake detection
"""

import cv2
import numpy as np
from PIL import Image
import logging
import os

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing pipeline for deepfake detection model
    """
    
    @staticmethod
    def load_image(image_path):
        """
        Load image from file path
        
        Args:
            image_path: str, path to image file
            
        Returns:
            numpy array: Image in BGR format or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            logger.info(f"Image loaded successfully: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    @staticmethod
    def load_image_from_bytes(image_bytes):
        """
        Load image from bytes (uploaded file)
        
        Args:
            image_bytes: bytes, image data
            
        Returns:
            numpy array: Image in BGR format
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info(f"Image loaded from bytes: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"Error loading image from bytes: {str(e)}")
            return None
    
    @staticmethod
    def normalize_image(image):
        """
        Normalize image to [0, 1] range
        
        Args:
            image: numpy array, image in BGR format
            
        Returns:
            numpy array: Normalized image (float32)
        """
        try:
            normalized = image.astype('float32') / 255.0
            logger.debug("Image normalized to [0, 1]")
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing image: {str(e)}")
            return None
    
    @staticmethod
    def preprocess_for_model(face_image):
        """
        Complete preprocessing pipeline for model input
        
        Args:
            face_image: numpy array, cropped face (224x224)
            
        Returns:
            numpy array: Preprocessed image ready for model (1, 224, 224, 3)
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Normalize
            normalized = rgb_image.astype('float32') / 255.0
            
            # Add batch dimension
            preprocessed = np.expand_dims(normalized, axis=0)
            
            logger.debug(f"Image preprocessed for model: {preprocessed.shape}")
            return preprocessed
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            return None


class VideoPreprocessor:
    """
    Video preprocessing for deepfake detection
    Extracts frames from video for analysis
    """
    
    @staticmethod
    def extract_frames(video_path, num_frames=10, method='uniform'):
        """
        Extract frames from video
        
        Args:
            video_path: str, path to video file
            num_frames: int, number of frames to extract
            method: str, 'uniform' or 'random' frame sampling
            
        Returns:
            list: List of frames (numpy arrays)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video info - Total frames: {total_frames}, FPS: {fps}")
            
            if total_frames == 0:
                logger.error("Video has no frames")
                cap.release()
                return []
            
            # Determine frame indices to extract
            if method == 'uniform':
                # Uniformly sample frames across video
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            else:  # random
                # Random sampling
                indices = np.random.choice(total_frames, min(num_frames, total_frames), replace=False)
                indices = np.sort(indices)
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frames.append(frame)
                    logger.debug(f"Extracted frame {idx}/{total_frames}")
                else:
                    logger.warning(f"Failed to read frame {idx}")
            
            cap.release()
            logger.info(f"Successfully extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames from video: {str(e)}")
            return []
    
    @staticmethod
    def extract_frames_from_bytes(video_bytes, num_frames=10, method='uniform'):
        """
        Extract frames from video bytes (uploaded file)
        
        Args:
            video_bytes: bytes, video data
            num_frames: int, number of frames to extract
            method: str, frame sampling method
            
        Returns:
            list: List of frames (numpy arrays)
        """
        try:
            # Save bytes to temporary file
            temp_path = 'temp_video.mp4'
            with open(temp_path, 'wb') as f:
                f.write(video_bytes)
            
            # Extract frames
            frames = VideoPreprocessor.extract_frames(temp_path, num_frames, method)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames from bytes: {str(e)}")
            return []
    
    @staticmethod
    def get_video_info(video_path):
        """
        Get video metadata
        
        Args:
            video_path: str, path to video file
            
        Returns:
            dict: Video information (fps, total_frames, duration, width, height)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
            
            cap.release()
            logger.info(f"Video info retrieved: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return None


def save_uploaded_file(file_bytes, filename, upload_dir='uploads'):
    """
    Save uploaded file to disk
    
    Args:
        file_bytes: bytes, file data
        filename: str, original filename
        upload_dir: str, directory to save file
        
    Returns:
        str: Path to saved file or None if failed
    """
    try:
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(file_bytes)
        
        logger.info(f"File saved: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return None
