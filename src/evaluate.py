import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from .data_loader import load_data

def evaluate(model_path='../models/saved_model/best_model.h5', 
            data_dir='../data/processed',
            img_size=(224, 224)):
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load class indices
    with open(os.path.join(os.path.dirname(model_path), 'class_indices.json'), 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Load validation data
    _, val_gen = load_data(data_dir, img_size, batch_size=32, val_split=0.2)
    
    # Evaluate
    results = model.evaluate(val_gen)
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
    
    # Predictions
    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, 
                               target_names=list(class_indices.keys())))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_indices.keys(), 
                yticklabels=class_indices.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../models/confusion_matrix.png')
    print("Confusion matrix saved to models/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()