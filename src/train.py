import os
import json
import argparse
import numpy as np
from datetime import datetime
from .model import create_model
from .data_loader import load_data
import tensorflow as tf

def train():
    # Configuration
    DATA_DIR = '../data/processed'
    MODEL_DIR = '../models/saved_model'
    LOG_DIR = '../models/training_logs'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Load data
    train_gen, val_gen = load_data(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    num_classes = len(train_gen.class_indices)
    
    # Create model
    model = create_model(num_classes)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=1e-6
        )
    ]
    
    # Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
    
    # Save class indices
    with open(os.path.join(MODEL_DIR, 'class_indices.json'), 'w') as f:
        json.dump(train_gen.class_indices, f)
    
    print("Training completed. Models saved to", MODEL_DIR)

if __name__ == "__main__":
    train()