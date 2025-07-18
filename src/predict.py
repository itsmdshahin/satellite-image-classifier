import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def predict(model_path, img_path, img_size=(224, 224)):
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load class indices
    with open(os.path.join(os.path.dirname(model_path), 'class_indices.json'), 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Preprocess image
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Prediction
    pred = model.predict(img_array)
    pred_class = idx_to_class[np.argmax(pred)]
    confidence = np.max(pred)
    
    # Visualization
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}\nConfidence: {confidence:.2%}")
    plt.axis('off')
    plt.savefig('../output/prediction_result.png')
    plt.show()
    
    return pred_class, confidence

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    args = parser.parse_args()
    
    pred_class, confidence = predict(args.model, args.image)
    print(f"Predicted class: {pred_class} with {confidence:.2%} confidence")