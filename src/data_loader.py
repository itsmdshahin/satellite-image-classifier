import tensorflow as tf
import albumentations as A
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def augment_image(image):
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(p=0.1),
        A.RandomGamma(p=0.2),
    ])
    return transform(image=image)['image']

def load_data(data_dir='../data/processed', img_size=(224, 224), batch_size=32, val_split=0.2):
    # Custom augmentation function for TensorFlow
    def tf_augment(image, label):
        image = tf.numpy_function(func=augment_image, 
                                  inp=[image], 
                                  Tout=tf.float32)
        image.set_shape((img_size[0], img_size[1], 3))
        return image, label

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Apply custom augmentations
    train_generator = train_generator.map(
        tf_augment, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return train_generator, val_generator