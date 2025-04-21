import os
import shutil
import random

# Define paths
source_dir = "RealWaste2"          # Folder with all images (before splitting)
target_dir = "Dataset_Split"       # Folder where split dataset will be saved

# Define split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create directories for train, val, test
for split in ["train", "val", "test"]:
    split_path = os.path.join(target_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Split images for each class
for class_name in sorted(os.listdir(source_dir)):
    class_path = os.path.join(source_dir, class_name)

    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        random.shuffle(images)  # Shuffle images for randomness

        # Compute split sizes
        num_images = len(images)
        train_count = int(num_images * train_ratio)
        val_count = int(num_images * val_ratio)
        
        # Assign images to sets
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        # Copy images to respective folders
        for split, img_list in zip(["train", "val", "test"], [train_images, val_images, test_images]):
            split_class_path = os.path.join(target_dir, split, class_name)
            os.makedirs(split_class_path, exist_ok=True)

            for img_name in img_list:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(split_class_path, img_name)
                shutil.copy(src, dst)

print("Dataset successfully split into train, validation, and test folders.")
