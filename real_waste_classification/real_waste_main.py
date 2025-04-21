import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from IPython.display import HTML

# Image data generators for training, validation, and testing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model architecture
from tensorflow.keras import layers, models

# Define constants for image size and batch size
IMAGE_SIZE = 256
BATCH_SIZE = 32

# Load the dataset from directory and prepare it for training
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "RealWaste",
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

# Counting the number of images in each class
dataset_path = "RealWaste" 
class_counts = {}

for class_name in sorted(os.listdir(dataset_path)):  
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        class_counts[class_name] = len(os.listdir(class_dir))

# Print class-wise image count
for class_name, count in class_counts.items():
    print(f"Class '{class_name}': {count} images")

# Print total number of images
print("\nTotal images:", sum(class_counts.values()))

# Define paths for dataset and target augmented dataset
source_dir = "RealWaste"  
target_dir = "RealWaste2" 
target_images_per_class = 921  # Target number of images per class after augmentation

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Random rotation within 30 degrees
    width_shift_range=0.2,  # Random horizontal shift by 20%
    height_shift_range=0.2,  # Random vertical shift by 20%
    shear_range=0.2,  # Random shear transformation
    zoom_range=0.3,  # Random zoom by 30%
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode="nearest"  # Fill missing pixels after transformation
)

# Create target directory structure if not exists
os.makedirs(target_dir, exist_ok=True)

for class_name in sorted(os.listdir(source_dir)):
    class_path = os.path.join(source_dir, class_name)
    target_class_path = os.path.join(target_dir, class_name)

    if os.path.isdir(class_path):
        os.makedirs(target_class_path, exist_ok=True)

        # Get list of images in the class
        images = [img for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(images)

        # Copy original images to new folder
        for img_name in images:
            shutil.copy(os.path.join(class_path, img_name), target_class_path)

        # Augment images if needed
        if num_images < target_images_per_class:
            num_to_generate = target_images_per_class - num_images
            print(f"Generating {num_to_generate} images for class '{class_name}'...")

            i = 0
            while i < num_to_generate:
                img_name = np.random.choice(images)  # Pick a random image
                img_path = os.path.join(class_path, img_name)

                # Load image and apply augmentation
                img = load_img(img_path)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                aug_iter = datagen.flow(img_array, batch_size=1)

                # Save augmented images
                for _ in range(min(num_to_generate - i, 5)):  # Generate in small batches
                    aug_img = next(aug_iter)[0]
                    aug_img = (aug_img * 255).astype("uint8")  # Convert back to uint8
                    aug_img_name = f"aug_{i}_{img_name}"
                    tf.keras.preprocessing.image.save_img(os.path.join(target_class_path, aug_img_name), aug_img)
                    i += 1


# Count and display the number of images in the augmented dataset
target_dataset_path = "RealWaste2"
class_counts = {}

for class_name in sorted(os.listdir(target_dataset_path)):  
    class_dir = os.path.join(target_dataset_path, class_name)
    if os.path.isdir(class_dir):
        class_counts[class_name] = len(os.listdir(class_dir))

# Print the number of images per class in the augmented dataset
print("Balanced dataset image counts:", class_counts)
print("\nTotal images:", sum(class_counts.values()))

# Load the newly created balanced dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "RealWaste2",
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE
)

# Check the length of the dataset
len(dataset)

# Display sample images from the dataset
for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())

# Display class names
class_names = dataset.class_names
class_names

# Plot sample images from the dataset
plt.figure (figsize=(10,10))
for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")

# Function to partition dataset into train, validation, and test sets
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
        
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

# Split dataset into train, validation, and test
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# Display the number of samples in each partition
print(len(train_ds), len(val_ds), len(test_ds))



# Function to show sample images from the dataset
def show_sample_images(ds, title, num_images=3, images_per_row=3):
    rows = (num_images + images_per_row - 1) // images_per_row  
    plt.figure(figsize=(images_per_row * 3, rows * 3)) 
    for image_batch, label_batch in ds.take(1): 
        for i in range(num_images):
            plt.subplot(rows, images_per_row, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[label_batch[i].numpy()])
            plt.axis("off")
    
    plt.suptitle(title)
    plt.show()

# Display sample images from the train, validation, and test sets
show_sample_images(train_ds, "Training Data")
show_sample_images(val_ds, "Validation Data")
show_sample_images(test_ds, "Test Data")

# Prepare dataset for efficient loading during training
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# Define image size and channels
IMAGE_SIZE = 256
CHANNELS = 3

# Define training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to [0, 1]
    horizontal_flip=True,  # Randomly flip images horizontally
    rotation_range=30,  # Random rotation within 30 degrees  
    width_shift_range=0.2,  # Random horizontal shift by 20%  
    height_shift_range=0.2,  # Random vertical shift by 20%
    zoom_range=0.3,  # Random zoom by 30%  
    shear_range=0.2  # Random shear transformation
)

# Generator for training set
train_generator = train_datagen.flow_from_directory(
    'Dataset_Split/train',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='sparse'
) 

# Define validation data augmentation
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2
)

# Generator for validation set
validation_generator = validation_datagen.flow_from_directory(
    'Dataset_Split/val',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='sparse'
) 

# Define test data augmentation
test_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2
)

# Generator for test set
test_generator = test_datagen.flow_from_directory(
    'Dataset_Split/test',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32,
    class_mode='sparse'
) 

# Input shape and number of classes
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 9

# Define model architecture
model = models.Sequential([
    layers.Input(shape=input_shape), 

    layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

# Print model summary
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=182,
    batch_size=32,
    validation_data=validation_generator,
    validation_steps=52,
    verbose=1,
    epochs=82,
)

# Evaluate the model on test data
scores = model.evaluate(test_generator)

# Print evaluation results
scores

# Print history of training
history

# Access history parameters
history.params

# View keys in the history object
history.history.keys()

# Check type of history loss
type(history.history['loss'])

# View loss history
len(history.history['loss'])

# Display first 5 entries in loss history
history.history['loss'][:5]

# Extract accuracy and loss values from history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation accuracy and loss
EPOCHS = 82

plt.figure (figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),acc, label='Training Accuracy')
plt.plot(range(EPOCHS),val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Predicting with the model
for image_batch, label_batch in test_generator:
    first_image = image_batch[0]
    first_label = int(label_batch[0]) 

    # Display first image to predict
    print("First image to predict:")
    plt.imshow(first_image)
    plt.axis('off') 
    plt.show()

    print("Actual label:", class_names[first_label])

    # Perform prediction on the batch
    batch_prediction = model.predict(image_batch) 
    print("Predicted label:", class_names[np.argmax(batch_prediction[0])])

    break

# Function to predict a single image
def predict (model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i])
    img_array = tf.expand_dims(img_array,0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)
    return predicted_class, confidence

# Display sample predictions
plt.figure(figsize = (15,15))
for images, labels in test_generator:
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i])

        # Predict and display result
        predicted_class, confidence = predict(model, images[i])
        actual_class = class_names[int(labels[i])]

        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")

        plt.axis("off")
    break

# Save the model
model.save("../waste2.h5")

# Load the saved model
model = tf.keras.models.load_model("../waste2.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimizations for the conversion
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("waste_model.tflite", "wb") as f:
    f.write(tflite_model)
