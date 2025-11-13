"""
Model Predictor Module
Handles model loading and inference for deepfake detection
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)


class DeepfakePredictor:
    """
    Predictor class for deepfake detection
    Loads trained model and performs inference
    """
    
    def __init__(self, model_path='model/xception_model.h5'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: str, path to trained model file
        """
        self.model_path = model_path
        self.model = None
        self.class_names = ['REAL', 'FAKE']
        self.load_model()
    
    def load_model(self):
        """
        Load trained model from disk
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict_single(self, preprocessed_image):
        """
        Predict on a single preprocessed image
        
        Args:
            preprocessed_image: numpy array, shape (1, 224, 224, 3), normalized to [0, 1]
            
        Returns:
            dict: Prediction results with label, confidence, and probabilities
        """
        try:
            # Make prediction
            predictions = self.model.predict(preprocessed_image, verbose=0)
            
            # Handle binary classification output (single sigmoid output)
            # Model was trained with: 0=fake, 1=real (reversed from typical)
            # predictions[0][0] is between 0 (FAKE) and 1 (REAL)
            real_probability = float(predictions[0][0])
            fake_probability = 1.0 - real_probability
            
            # Determine predicted class
            if real_probability > 0.5:
                label = 'REAL'
                confidence = real_probability * 100
            else:
                label = 'FAKE'
                confidence = fake_probability * 100
            
            result = {
                'label': label,
                'confidence': round(confidence, 2),
                'probabilities': {
                    'REAL': round(real_probability * 100, 2),
                    'FAKE': round(fake_probability * 100, 2)
                }
            }
            
            logger.info(f"Prediction: {label} ({confidence:.2f}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None
    
    def predict_batch(self, preprocessed_images):
        """
        Predict on a batch of preprocessed images
        
        Args:
            preprocessed_images: numpy array, shape (n, 224, 224, 3)
            
        Returns:
            list: List of prediction results
        """
        try:
            # Make predictions
            predictions = self.model.predict(preprocessed_images, verbose=0)
            
            results = []
            for pred in predictions:
                prob_real = float(pred[0])
                prob_fake = float(pred[1])
                predicted_class = np.argmax(pred)
                label = self.class_names[predicted_class]
                confidence = float(pred[predicted_class]) * 100
                
                result = {
                    'label': label,
                    'confidence': round(confidence, 2),
                    'probabilities': {
                        'REAL': round(prob_real * 100, 2),
                        'FAKE': round(prob_fake * 100, 2)
                    }
                }
                results.append(result)
            
            logger.info(f"Batch prediction completed for {len(results)} images")
            return results
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            return None
    
    def predict_video_frames(self, preprocessed_frames):
        """
        Predict on multiple video frames and return aggregated result
        
        Args:
            preprocessed_frames: list of numpy arrays, each shape (1, 224, 224, 3)
            
        Returns:
            dict: Aggregated prediction result with average confidence
        """
        try:
            if not preprocessed_frames:
                logger.warning("No frames to predict")
                return None
            
            # Combine frames into single batch
            batch = np.vstack(preprocessed_frames)
            
            # Make predictions
            predictions = self.model.predict(batch, verbose=0)
            
            # Calculate average probabilities
            avg_prob_real = float(np.mean(predictions[:, 0]))
            avg_prob_fake = float(np.mean(predictions[:, 1]))
            
            # Determine overall label based on average
            if avg_prob_fake > avg_prob_real:
                label = 'FAKE'
                confidence = avg_prob_fake * 100
            else:
                label = 'REAL'
                confidence = avg_prob_real * 100
            
            # Calculate consistency (standard deviation)
            std_dev = float(np.std(predictions[:, 1]))  # Std of FAKE probabilities
            consistency_score = max(0, 100 - (std_dev * 100))
            
            result = {
                'label': label,
                'confidence': round(confidence, 2),
                'probabilities': {
                    'REAL': round(avg_prob_real * 100, 2),
                    'FAKE': round(avg_prob_fake * 100, 2)
                },
                'frames_analyzed': len(preprocessed_frames),
                'consistency_score': round(consistency_score, 2),
                'frame_predictions': []
            }
            
            # Add individual frame predictions
            for i, pred in enumerate(predictions):
                frame_result = {
                    'frame_number': i + 1,
                    'prob_fake': round(float(pred[1]) * 100, 2)
                }
                result['frame_predictions'].append(frame_result)
            
            logger.info(f"Video prediction: {label} ({confidence:.2f}%) - {len(preprocessed_frames)} frames")
            return result
            
        except Exception as e:
            logger.error(f"Error during video frame prediction: {str(e)}")
            return None
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return None
        
        return {
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params(),
            'class_names': self.class_names
        }


# Singleton instance
_predictor_instance = None


def get_predictor(model_path='model/xception_model.h5'):
    """
    Get or create singleton predictor instance
    
    Args:
        model_path: str, path to model file
        
    Returns:
        DeepfakePredictor: Singleton predictor instance
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DeepfakePredictor(model_path)
    return _predictor_instance
