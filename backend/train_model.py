"""
Model Training Script for DeepGuard Deepfake Detection
Uses XceptionNet (pretrained on ImageNet) fine-tuned for binary classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepfakeModel:
    """
    Deepfake detection model based on XceptionNet
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Initialize model architecture
        
        Args:
            input_shape: tuple, input image shape (height, width, channels)
            num_classes: int, number of output classes (2 for binary: real/fake)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self, fine_tune_layers=50):
        """
        Build XceptionNet-based model for deepfake detection
        
        Args:
            fine_tune_layers: int, number of top layers to unfreeze for fine-tuning
            
        Returns:
            keras.Model: Compiled model
        """
        logger.info("Building XceptionNet model...")
        
        # Load pretrained XceptionNet (without top classification layer)
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=outputs)
        
        logger.info(f"Model built with {len(self.model.layers)} layers")
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile model with optimizer and loss function
        
        Args:
            learning_rate: float, initial learning rate
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        logger.info(f"Model compiled with learning rate: {learning_rate}")
    
    def unfreeze_layers(self, num_layers=50):
        """
        Unfreeze top layers for fine-tuning
        
        Args:
            num_layers: int, number of layers to unfreeze from the end
        """
        # Get the base model (Xception)
        base_model = self.model.layers[0]
        
        # Unfreeze the top layers
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
        
        logger.info(f"Unfroze top {num_layers} layers for fine-tuning")
    
    def save_model(self, filepath='model/xception_model.h5'):
        """
        Save trained model
        
        Args:
            filepath: str, path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath='model/xception_model.h5'):
        """
        Load saved model
        
        Args:
            filepath: str, path to model file
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from: {filepath}")


def create_data_generators(train_dir, val_dir, batch_size=32, img_size=(224, 224)):
    """
    Create data generators for training and validation
    
    Args:
        train_dir: str, path to training data directory
        val_dir: str, path to validation data directory
        batch_size: int, batch size for training
        img_size: tuple, target image size
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=0.15,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    logger.info(f"Data generators created - Train samples: {train_generator.samples}, Val samples: {val_generator.samples}")
    return train_generator, val_generator


def train_model(train_dir, val_dir, epochs=50, batch_size=32, model_save_path='model/xception_model.h5'):
    """
    Complete training pipeline
    
    Args:
        train_dir: str, training data directory (should contain 'real' and 'fake' subdirectories)
        val_dir: str, validation data directory
        epochs: int, number of training epochs
        batch_size: int, batch size
        model_save_path: str, path to save trained model
        
    Returns:
        keras.callbacks.History: Training history
    """
    logger.info("Starting model training pipeline...")
    
    # Create data generators
    train_gen, val_gen = create_data_generators(train_dir, val_dir, batch_size)
    
    # Build model
    deepfake_model = DeepfakeModel()
    model = deepfake_model.build_model()
    deepfake_model.compile_model(learning_rate=0.001)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr]
    
    # Phase 1: Train only the top layers
    logger.info("Phase 1: Training top layers only...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs // 2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen layers
    logger.info("Phase 2: Fine-tuning with unfrozen layers...")
    deepfake_model.unfreeze_layers(num_layers=50)
    deepfake_model.compile_model(learning_rate=0.0001)  # Lower learning rate for fine-tuning
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs // 2,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed!")
    logger.info(f"Best model saved at: {model_save_path}")
    
    return history1, history2


def create_simple_model(save_path='model/xception_model.h5'):
    """
    Create and save a simple model structure for testing
    (Use this if you don't have training data yet)
    
    Args:
        save_path: str, path to save model
    """
    logger.info("Creating simple model for testing...")
    
    deepfake_model = DeepfakeModel()
    model = deepfake_model.build_model()
    deepfake_model.compile_model()
    
    # Save untrained model structure
    deepfake_model.save_model(save_path)
    logger.info(f"Untrained model saved to: {save_path}")


if __name__ == "__main__":
    """
    Main training script
    
    Directory structure expected:
    data/
        train/
            real/
                image1.jpg
                image2.jpg
                ...
            fake/
                image1.jpg
                image2.jpg
                ...
        val/
            real/
                image1.jpg
                ...
            fake/
                image1.jpg
                ...
    """
    
    # Configuration
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    MODEL_SAVE_PATH = 'model/xception_model.h5'
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Check if data directories exist
    if os.path.exists(TRAIN_DIR) and os.path.exists(VAL_DIR):
        logger.info("Training data found. Starting training...")
        train_model(
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            model_save_path=MODEL_SAVE_PATH
        )
    else:
        logger.warning("Training data not found. Creating model structure for testing...")
        logger.warning(f"Please prepare data in the following structure:")
        logger.warning(f"  {TRAIN_DIR}/real/ and {TRAIN_DIR}/fake/")
        logger.warning(f"  {VAL_DIR}/real/ and {VAL_DIR}/fake/")
        create_simple_model(MODEL_SAVE_PATH)
