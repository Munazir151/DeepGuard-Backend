"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Generates heatmaps showing which regions of the image influenced the model's decision
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM visualization for deep learning models
    Shows which parts of the image are most important for the prediction
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: keras.Model, trained model
            layer_name: str, name of the convolutional layer to visualize
                       If None, uses the last convolutional layer
        """
        self.model = model
        self.layer_name = layer_name
        
        if self.layer_name is None:
            # Find last convolutional layer
            self.layer_name = self._find_last_conv_layer()
        
        logger.info(f"GradCAM initialized with layer: {self.layer_name}")
    
    def _find_last_conv_layer(self):
        """
        Find the name of the last convolutional layer in the model
        
        Returns:
            str: Layer name
        """
        for layer in reversed(self.model.layers):
            # Check if it's a convolutional layer
            if len(layer.output_shape) == 4:
                return layer.name
        
        # If no conv layer found in main model, check base model
        if hasattr(self.model.layers[0], 'layers'):
            for layer in reversed(self.model.layers[0].layers):
                if len(layer.output_shape) == 4:
                    return layer.name
        
        raise ValueError("Could not find a convolutional layer in the model")
    
    def compute_heatmap(self, image, class_idx=None, alpha=0.4):
        """
        Compute Grad-CAM heatmap for an image
        
        Args:
            image: numpy array, preprocessed image (1, 224, 224, 3)
            class_idx: int, class index to visualize (None = predicted class)
            alpha: float, transparency for overlay
            
        Returns:
            numpy array: Heatmap image
        """
        try:
            # Create a model that outputs both predictions and conv layer output
            grad_model = keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
            )
            
            # Compute gradient of predicted class with respect to feature maps
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0])
                
                class_output = predictions[:, class_idx]
            
            # Compute gradients
            grads = tape.gradient(class_output, conv_outputs)
            
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight feature maps by gradients
            conv_outputs = conv_outputs[0]
            pooled_grads = pooled_grads.numpy()
            conv_outputs = conv_outputs.numpy()
            
            for i in range(len(pooled_grads)):
                conv_outputs[:, :, i] *= pooled_grads[i]
            
            # Create heatmap
            heatmap = np.mean(conv_outputs, axis=-1)
            
            # Normalize heatmap
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) != 0:
                heatmap = heatmap / np.max(heatmap)
            
            logger.info("Grad-CAM heatmap computed successfully")
            return heatmap
            
        except Exception as e:
            logger.error(f"Error computing Grad-CAM heatmap: {str(e)}")
            return None
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: numpy array, computed heatmap
            original_image: numpy array, original image (can be any size)
            alpha: float, transparency (0=only image, 1=only heatmap)
            colormap: int, OpenCV colormap
            
        Returns:
            numpy array: Image with heatmap overlay
        """
        try:
            # Resize heatmap to match image size
            if original_image.shape[0] == 1:
                original_image = original_image[0]
            
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # Convert heatmap to uint8
            heatmap = np.uint8(255 * heatmap)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap, colormap)
            
            # Convert original image to uint8 if needed
            if original_image.max() <= 1.0:
                original_image = np.uint8(255 * original_image)
            
            # Ensure image is in BGR format
            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            elif original_image.shape[2] == 4:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2BGR)
            
            # Overlay heatmap on image
            overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
            
            logger.info("Heatmap overlaid on image successfully")
            return overlaid
            
        except Exception as e:
            logger.error(f"Error overlaying heatmap: {str(e)}")
            return None
    
    def generate_gradcam(self, image, original_image=None, class_idx=None, alpha=0.4):
        """
        Complete Grad-CAM pipeline: compute heatmap and overlay on image
        
        Args:
            image: numpy array, preprocessed image for model (1, 224, 224, 3)
            original_image: numpy array, original image for overlay (optional)
            class_idx: int, class to visualize (None = predicted class)
            alpha: float, overlay transparency
            
        Returns:
            dict: Results containing heatmap, overlay, and metadata
        """
        try:
            # Compute heatmap
            heatmap = self.compute_heatmap(image, class_idx, alpha)
            
            if heatmap is None:
                return None
            
            # Use preprocessed image if no original provided
            if original_image is None:
                original_image = image[0]
            
            # Create overlay
            overlaid = self.overlay_heatmap(heatmap, original_image, alpha)
            
            if overlaid is None:
                return None
            
            result = {
                'heatmap': heatmap,
                'overlay': overlaid,
                'layer_name': self.layer_name,
                'class_idx': int(class_idx) if class_idx is not None else None
            }
            
            logger.info("Grad-CAM visualization generated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {str(e)}")
            return None
    
    def save_gradcam(self, result, output_path):
        """
        Save Grad-CAM visualization to file
        
        Args:
            result: dict, output from generate_gradcam()
            output_path: str, path to save image
        """
        try:
            if result is None or 'overlay' not in result:
                logger.error("Invalid Grad-CAM result")
                return False
            
            cv2.imwrite(output_path, result['overlay'])
            logger.info(f"Grad-CAM saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving Grad-CAM: {str(e)}")
            return False


def create_gradcam_visualization(model, image, original_image=None, layer_name=None):
    """
    Convenience function to create Grad-CAM visualization
    
    Args:
        model: keras.Model, trained model
        image: numpy array, preprocessed image (1, 224, 224, 3)
        original_image: numpy array, original image for overlay
        layer_name: str, layer to visualize (None = last conv layer)
        
    Returns:
        dict: Grad-CAM results
    """
    try:
        gradcam = GradCAM(model, layer_name)
        result = gradcam.generate_gradcam(image, original_image)
        return result
    except Exception as e:
        logger.error(f"Error creating Grad-CAM visualization: {str(e)}")
        return None
