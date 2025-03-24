from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data():
    # Define paths to train and test directories
    train_dir = 'data/train'
    test_dir = 'data/test'

    # Create the ImageDataGenerator instance for training and testing
    train_datagen = ImageDataGenerator(rescale=1./255)  # Normalize images
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators to load the images
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),  # Resize images to 48x48 pixels
        batch_size=32,         # Load 32 images per batch
        class_mode='categorical'  # Use one-hot encoding for labels
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(48, 48),  # Resize images to 48x48 pixels
        batch_size=32,         # Load 32 images per batch
        class_mode='categorical'  # Use one-hot encoding for labels
    )

    return train_generator, test_generator
