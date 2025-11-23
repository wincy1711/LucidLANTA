#!/usr/bin/env python3
"""
TensorFlow training example for LANTA cluster
Optimized for GPU training with distributed strategies
"""

import os
import time
import argparse
import logging
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tensorflow_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_gpu_memory_growth():
    """Configure GPU memory growth"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.error(f"GPU memory growth setup failed: {e}")

def create_cnn_model(input_shape, num_classes):
    """Create a CNN model for CIFAR-10"""
    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_data_augmentation():
    """Create data augmentation pipeline"""
    
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    
    return data_augmentation

def preprocess_data(images, labels, data_augmentation=None):
    """Preprocess data with normalization and augmentation"""
    
    # Normalize images
    images = tf.cast(images, tf.float32) / 255.0
    
    # Apply data augmentation if provided
    if data_augmentation is not None:
        images = data_augmentation(images)
    
    return images, labels

def create_datasets(data_path, batch_size, buffer_size=10000):
    """Create training and validation datasets"""
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Create data augmentation
    data_augmentation = create_data_augmentation()
    
    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.map(
        lambda x, y: preprocess_data(x, y, data_augmentation),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.map(
        lambda x, y: preprocess_data(x, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset

def train_model(model, train_dataset, val_dataset, epochs, learning_rate, model_dir, log_dir):
    """Train the model with callbacks and monitoring"""
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        ),
        keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv'),
            separator=',',
            append=False
        )
    ]
    
    # Train model
    logger.info('Starting model training...')
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    logger.info(f'Training completed in {training_time:.2f} seconds')
    
    return history

def evaluate_model(model, test_dataset):
    """Evaluate model performance"""
    
    logger.info('Evaluating model...')
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    
    logger.info(f'Test Loss: {test_loss:.4f}')
    logger.info(f'Test Accuracy: {test_accuracy:.4f}')
    
    return test_loss, test_accuracy

def save_training_history(history, output_dir):
    """Save training history to JSON"""
    
    history_dict = history.history
    history_dict['epochs'] = len(history.history['loss'])
    
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    logger.info('Training history saved')

def main():
    parser = argparse.ArgumentParser(description='TensorFlow CIFAR-10 Training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument('--model-dir', type=str, default='./models', help='model directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='log directory')
    
    args = parser.parse_args()
    
    # Setup GPU
    setup_gpu_memory_growth()
    
    # Log GPU information
    gpus = tf.config.experimental.list_physical_devices('GPU')
    logger.info(f'Available GPUs: {len(gpus)}')
    for i, gpu in enumerate(gpus):
        logger.info(f'GPU {i}: {gpu.name}')
    
    # Create directories
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    train_dataset, test_dataset = create_datasets(
        args.data_path, args.batch_size
    )
    
    # Log dataset information
    logger.info(f'Training batches: {len(train_dataset)}')
    logger.info(f'Test batches: {len(test_dataset)}')
    
    # Create model
    model = create_cnn_model((32, 32, 3), 10)
    
    # Log model summary
    model.summary(print_fn=logger.info)
    
    # Train model
    history = train_model(
        model, train_dataset, test_dataset,
        args.epochs, args.learning_rate,
        args.model_dir, args.log_dir
    )
    
    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, test_dataset)
    
    # Save training history
    save_training_history(history, args.log_dir)
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.h5')
    model.save(final_model_path)
    logger.info(f'Final model saved to {final_model_path}')
    
    # Log final results
    logger.info('='*50)
    logger.info('TRAINING COMPLETED')
    logger.info('='*50)
    logger.info(f'Final Test Accuracy: {test_accuracy:.4f}')
    logger.info(f'Best Validation Accuracy: {max(history.history["val_accuracy"]):.4f}')
    logger.info(f'Total Epochs: {len(history.history["loss"])}')

if __name__ == '__main__':
    main()