"""
Frequency Domain Analysis Module
Implements FFT and DCT analysis to detect deepfake artifacts
"""

import cv2
import numpy as np
from scipy import fftpack
import logging

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """
    Frequency domain analysis for detecting deepfake artifacts
    Uses FFT (Fast Fourier Transform) and DCT (Discrete Cosine Transform)
    """
    
    @staticmethod
    def compute_fft(image):
        """
        Compute 2D FFT (Fast Fourier Transform) of image
        
        Args:
            image: numpy array, grayscale or color image
            
        Returns:
            numpy array: FFT magnitude spectrum
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Compute FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            
            # Compute magnitude spectrum
            magnitude_spectrum = np.abs(fft_shift)
            
            logger.debug("FFT computed successfully")
            return magnitude_spectrum
            
        except Exception as e:
            logger.error(f"Error computing FFT: {str(e)}")
            return None
    
    @staticmethod
    def compute_dct(image):
        """
        Compute 2D DCT (Discrete Cosine Transform) of image
        
        Args:
            image: numpy array, grayscale or color image
            
        Returns:
            numpy array: DCT coefficients
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Convert to float
            gray_float = gray.astype(np.float32)
            
            # Compute DCT
            dct = fftpack.dct(fftpack.dct(gray_float.T, norm='ortho').T, norm='ortho')
            
            logger.debug("DCT computed successfully")
            return dct
            
        except Exception as e:
            logger.error(f"Error computing DCT: {str(e)}")
            return None
    
    @staticmethod
    def extract_frequency_features(image):
        """
        Extract frequency domain features for deepfake detection
        
        Args:
            image: numpy array, face image
            
        Returns:
            dict: Frequency features including statistics from FFT and DCT
        """
        try:
            features = {}
            
            # Compute FFT
            fft_magnitude = FrequencyAnalyzer.compute_fft(image)
            if fft_magnitude is not None:
                # FFT statistics
                features['fft_mean'] = np.mean(fft_magnitude)
                features['fft_std'] = np.std(fft_magnitude)
                features['fft_max'] = np.max(fft_magnitude)
                
                # High frequency energy (outer regions)
                h, w = fft_magnitude.shape
                center_h, center_w = h // 2, w // 2
                radius = min(h, w) // 4
                
                mask = np.zeros_like(fft_magnitude)
                y, x = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
                mask[dist_from_center > radius] = 1
                
                high_freq_energy = np.sum(fft_magnitude * mask)
                total_energy = np.sum(fft_magnitude)
                features['high_freq_ratio'] = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # Compute DCT
            dct_coeffs = FrequencyAnalyzer.compute_dct(image)
            if dct_coeffs is not None:
                # DCT statistics
                features['dct_mean'] = np.mean(np.abs(dct_coeffs))
                features['dct_std'] = np.std(dct_coeffs)
                features['dct_max'] = np.max(np.abs(dct_coeffs))
                
                # Low frequency energy (top-left region)
                h, w = dct_coeffs.shape
                low_freq_region = dct_coeffs[:h//4, :w//4]
                features['dct_low_freq_energy'] = np.sum(np.abs(low_freq_region))
            
            logger.info(f"Extracted {len(features)} frequency features")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting frequency features: {str(e)}")
            return {}
    
    @staticmethod
    def compute_frequency_score(image):
        """
        Compute a single frequency anomaly score
        Higher score suggests more likely to be fake
        
        Args:
            image: numpy array, face image
            
        Returns:
            float: Frequency anomaly score [0, 1]
        """
        try:
            features = FrequencyAnalyzer.extract_frequency_features(image)
            
            if not features:
                return 0.5  # Neutral score if features couldn't be extracted
            
            # Deepfakes often have:
            # - Lower high-frequency energy (due to compression/smoothing)
            # - Different DCT coefficient distributions
            
            # Normalize and combine features (simple weighted combination)
            score = 0.0
            weight_sum = 0.0
            
            # High frequency ratio (lower = more suspicious)
            if 'high_freq_ratio' in features:
                # Invert: lower ratio = higher score
                hf_score = 1.0 - min(features['high_freq_ratio'] * 10, 1.0)
                score += hf_score * 0.4
                weight_sum += 0.4
            
            # DCT low frequency energy (normalized)
            if 'dct_low_freq_energy' in features:
                # Normalize to [0, 1] range (empirical threshold)
                dct_score = min(features['dct_low_freq_energy'] / 100000.0, 1.0)
                score += dct_score * 0.3
                weight_sum += 0.3
            
            # FFT statistics variation
            if 'fft_std' in features:
                fft_score = min(features['fft_std'] / 10000.0, 1.0)
                score += fft_score * 0.3
                weight_sum += 0.3
            
            final_score = score / weight_sum if weight_sum > 0 else 0.5
            logger.info(f"Frequency anomaly score: {final_score:.3f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error computing frequency score: {str(e)}")
            return 0.5
    
    @staticmethod
    def visualize_frequency_spectrum(image, output_path=None):
        """
        Create visualization of frequency spectrum
        
        Args:
            image: numpy array, input image
            output_path: str, optional path to save visualization
            
        Returns:
            numpy array: Visualization image
        """
        try:
            fft_magnitude = FrequencyAnalyzer.compute_fft(image)
            
            if fft_magnitude is None:
                return None
            
            # Log scale for better visualization
            magnitude_log = np.log(1 + fft_magnitude)
            
            # Normalize to [0, 255]
            magnitude_normalized = cv2.normalize(
                magnitude_log, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            
            # Apply colormap
            colored = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)
            
            if output_path:
                cv2.imwrite(output_path, colored)
                logger.info(f"Frequency spectrum saved to: {output_path}")
            
            return colored
            
        except Exception as e:
            logger.error(f"Error visualizing frequency spectrum: {str(e)}")
            return None


def analyze_image_frequency(image):
    """
    Convenience function for frequency analysis
    
    Args:
        image: numpy array, face image
        
    Returns:
        dict: Analysis results including score and features
    """
    analyzer = FrequencyAnalyzer()
    
    score = analyzer.compute_frequency_score(image)
    features = analyzer.extract_frequency_features(image)
    
    return {
        'frequency_score': score,
        'features': features
    }
