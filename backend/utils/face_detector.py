"""
Face Detection Module using MTCNN
Detects and crops faces from images for deepfake detection
"""

import cv2
import numpy as np
from mtcnn import MTCNN
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks)
    Provides robust face detection and extraction
    """
    
    def __init__(self):
        """Initialize MTCNN face detector"""
        try:
            self.detector = MTCNN()
            logger.info("MTCNN face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {str(e)}")
            raise
    
    def detect_face(self, image):
        """
        Detect the largest face in an image
        
        Args:
            image: numpy array (BGR format from OpenCV)
            
        Returns:
            dict: Detection result with 'box' and 'confidence' or None if no face found
        """
        try:
            # Convert BGR to RGB (MTCNN expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector.detect_faces(rgb_image)
            
            if not detections:
                logger.warning("No face detected in image")
                return None
            
            # Select the face with highest confidence
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            logger.info(f"Face detected with confidence: {best_detection['confidence']:.2f}")
            return best_detection
            
        except Exception as e:
            logger.error(f"Error during face detection: {str(e)}")
            return None
    
    def extract_face(self, image, required_size=(224, 224), padding=0.2):
        """
        Detect and extract face from image with padding
        
        Args:
            image: numpy array (BGR format)
            required_size: tuple, target size for face image (width, height)
            padding: float, padding ratio around detected face (0.2 = 20% padding)
            
        Returns:
            numpy array: Cropped and resized face image or None if no face found
        """
        try:
            detection = self.detect_face(image)
            
            if detection is None:
                return None
            
            # Extract bounding box
            x, y, width, height = detection['box']
            
            # Add padding
            pad_w = int(width * padding)
            pad_h = int(height * padding)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + width + pad_w)
            y2 = min(image.shape[0], y + height + pad_h)
            
            # Extract face region
            face = image[y1:y2, x1:x2]
            
            # Resize to required size
            face_resized = cv2.resize(face, required_size, interpolation=cv2.INTER_AREA)
            
            logger.info(f"Face extracted and resized to {required_size}")
            return face_resized
            
        except Exception as e:
            logger.error(f"Error during face extraction: {str(e)}")
            return None
    
    def extract_multiple_faces(self, image, required_size=(224, 224), padding=0.2):
        """
        Extract all detected faces from an image
        
        Args:
            image: numpy array (BGR format)
            required_size: tuple, target size for face images
            padding: float, padding ratio around detected faces
            
        Returns:
            list: List of cropped face images
        """
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb_image)
            
            if not detections:
                logger.warning("No faces detected in image")
                return []
            
            faces = []
            for detection in detections:
                x, y, width, height = detection['box']
                
                # Add padding
                pad_w = int(width * padding)
                pad_h = int(height * padding)
                
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(image.shape[1], x + width + pad_w)
                y2 = min(image.shape[0], y + height + pad_h)
                
                # Extract and resize face
                face = image[y1:y2, x1:x2]
                face_resized = cv2.resize(face, required_size, interpolation=cv2.INTER_AREA)
                faces.append(face_resized)
            
            logger.info(f"Extracted {len(faces)} faces from image")
            return faces
            
        except Exception as e:
            logger.error(f"Error extracting multiple faces: {str(e)}")
            return []


# Singleton instance for reuse
_face_detector_instance = None


def get_face_detector():
    """
    Get or create singleton face detector instance
    
    Returns:
        FaceDetector: Singleton face detector instance
    """
    global _face_detector_instance
    if _face_detector_instance is None:
        _face_detector_instance = FaceDetector()
    return _face_detector_instance
