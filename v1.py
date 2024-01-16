import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from keras_visualizer import visualizer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

os.environ["PATH"] += os.pathsep + 'graphviz'

# simple, default, xs, xl
model = "default"

# delete model saves
reset = True

image_size = (100, 100)
batch_size = 64

# epochs
epochs = 1

# model size

#model_size = [256,512,728]
#model_size = [256,728]
model_size = [256]
#
#

training_dir = 'training'

num_classes = len([name for name in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, name))])
print(f'Found {num_classes} classes.')

if reset == True:
    for file in os.listdir():
        if file.endswith('.keras'):
            os.remove(file)


#region 1. Filter out corrupted images

print('1. Removing corrupted images')

def remove_corrupt_images():
    num_skipped = 0
    for folder_name in tqdm(os.listdir(training_dir), desc="Folders"):
        if os.path.isdir(os.path.join(training_dir, folder_name)):
            folder_path = os.path.join(training_dir, folder_name)
            
            # Loop through all files in the current folder
            for fname in tqdm(os.listdir(folder_path), desc=f"Files in {folder_name}", leave=False):
                fpath = os.path.join(folder_path, fname)
                
                try:
                    fobj = open(fpath, "rb")
                    is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
                finally:
                    fobj.close()

                if not is_jfif:
                    num_skipped += 1
                    # Delete corrupted image
                    os.remove(fpath)

    print(f"Total removed files: {num_skipped}")

# remove_corrupt_images()
#endregion

#region 2. Generate a dataset">

print('2. Generating dataset')

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "training",
    validation_split=0.5,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# List existing folder names (class names)
#existing_class_names = [name for name in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, name))]

# Dynamic mapping to new class names. For this example, I'm just appending "_new" to existing names. Replace this to suit your needs.
#class_name_mapping = {name: name + "_new" for name in existing_class_names}

# Update class names in dataset
#train_ds.class_names = [class_name_mapping.get(name, name) for name in train_ds.class_names]
#val_ds.class_names = [class_name_mapping.get(name, name) for name in val_ds.class_names]


#endregion

#region 3. Visualize the data

print('3. Visualize the data')

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()
#endregion

#region 4. Using image data augmentation

print('4. Using image data augmentation')

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

# Show what it looks like

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
#endregion

#region 5. Standardize the data
# Our image are already in a standard size (180x180), as they are being yielded as contiguous float32 batches by our dataset. 
# However, their RGB channel values are in the [0, 255] range. This is not ideal for a neural network; 
# in general you should seek to make your input values small. 
# Here, we will standardize values to be in the [0, 1] by using a Rescaling layer at the start of our model.

#endregion

#region 6. Two options to preprocess the data, there are two ways you could be using the data_augmentation preprocessor:

print('6. Preprocessing')

# Option 1 (CPU): Make it part of the model:
#inputs = keras.Input(shape=input_shape)
#x = data_augmentation(inputs)
#x = layers.Rescaling(1./255)(x)

# Option 2 (CPU): apply it to the dataset, so as to obtain a dataset that yields batches of augmented images, like this:
augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))
#endregion

#region 7. Configure the dataset for performance

print('7. Configure the dataset for performance')

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
#endregion

#region 8. Build a model

print('8. Building model')

def make_model_default(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in model_size:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    units = num_classes
    
    # Fully Connected (Dense) Layer before output
    x = layers.Dense(512, activation='relu')(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def make_model_xl(input_shape, num_classes):
    # Input layer: Specify the shape of the input image
    inputs = keras.Input(shape=input_shape)

    # Entry block: Rescale, Conv2D, BatchNorm, and ReLU activation
    x = layers.Rescaling(1.0 / 255)(inputs)  # Rescale input to [0, 1]
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)  # Convolutional layer
    x = layers.BatchNormalization()(x)  # Batch normalization
    x = layers.Activation("relu")(x)  # ReLU activation
    previous_block_activation = x # Save this tensor for residual connections later
    
    # Adding Spatial Dropout
    x = layers.SpatialDropout2D(0.3)(x)

    # First block with 128 filters
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    # Adding Gaussian Noise for regularization
    x = layers.GaussianNoise(0.1)(x)

    # Project residual and add it back to x
    residual = layers.Conv2D(128, 1, strides=2, padding="same")(previous_block_activation)
    x = layers.add([x, residual])
    previous_block_activation = x # Update for next residual connection

    # Second block with 256 filters
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Attention Mechanism
    query = layers.Dense(256)(x)
    value = layers.Dense(256)(x)
    attention = layers.AdditiveAttention()([query, value])
    x = layers.add([x, attention])

    # Project residual and add it back to x
    residual = layers.Conv2D(256, 1, strides=2, padding="same")(previous_block_activation)
    x = layers.add([x, residual])

    # Final Conv and Global average Pooling
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully Connected (Dense) Layer before output
    x = layers.Dense(256, activation='relu')(x)
    
    # Second Dense Layer
    x = layers.Dense(128, activation='relu')(x)
    
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

def make_model_simple(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in model_size:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


# Check for existing model
model_files = [file for file in os.listdir() if file.endswith('.keras')]

if model_files:
    # Extract numbers and find the highest one
    model_numbers = [int(file.split("_")[2]) for file in model_files]  # Assuming file name is in the format "save_at_{epoch}_.keras"
    highest_number = max(model_numbers)
    
    # Create model file name with highest number
    model_file = f"save_at_{highest_number}_.keras"
    
    print(f"Loading existing model from {model_file} saved at epoch {highest_number}")
    model = keras.models.load_model(model_file)
    
    initial_epoch = highest_number
    
    # If you want to continue training from the next epoch
    #initial_epoch = highest_number + 1
else:
    # Existing code for building the model
    if model == 'simple':
        model = make_model_simple(input_shape=image_size + (3,), num_classes=num_classes)
    elif model == 'xl':
        model = make_model_xl(input_shape=image_size + (3,), num_classes=num_classes)
    else:
        model = make_model_default(input_shape=image_size + (3,), num_classes=num_classes)
    
    initial_epoch = 0


keras.utils.plot_model(model, show_shapes=True)

# save a view of the model
visualizer_settings = {
    'MAX_NEURONS': 64,
    'INPUT_DENSE_COLOR': 'teal',
    'HIDDEN_DENSE_COLOR': 'gray',
    'OUTPUT_DENSE_COLOR': 'crimson'
}


visualizer(model, file_name='model_visualized', file_format='pdf', settings=visualizer_settings)
#endregion

#9. Train the model

print('9. Training the model')

class FullBatchHistory(Callback):
    def on_train_begin(self, logs=None):
        self.train_batch_acc = []
        self.train_batch_loss = []
        self.val_batch_acc = []
        self.val_batch_loss = []
        
    def on_batch_end(self, batch, logs=None):
        self.train_batch_acc.append(logs.get('accuracy'))
        self.train_batch_loss.append(logs.get('loss'))
        
    def on_test_batch_end(self, batch, logs=None):
        self.val_batch_acc.append(logs.get('accuracy'))
        self.val_batch_loss.append(logs.get('loss'))



callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}_.keras"),
    FullBatchHistory()
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",  # or "categorical_crossentropy"
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
    initial_epoch=initial_epoch,
)

model.summary()



# 9.1 show training info

def display_epoch_info():
    # Get accuracy data
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Create epochs range
    epochs_range = range(epochs)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()


full_batch_history = callbacks[-1]  # Assuming FullBatchHistory is the last in your list

plt.figure(figsize=(12, 6))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(full_batch_history.train_batch_acc, label='Training Accuracy')
plt.plot(full_batch_history.val_batch_acc, label='Validation Accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(full_batch_history.train_batch_loss, label='Training Loss')
plt.plot(full_batch_history.val_batch_loss, label='Validation Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()

plt.suptitle('Batch-wise Training and Validation Metrics')
plt.show()


# 9.2 save the model



## 10. Run interference on new data
#img = keras.utils.load_img(
#    "training/Cat/30.jpg", target_size=image_size
#)
#plt.imshow(img)

#img_array = keras.utils.img_to_array(img)
#img_array = tf.expand_dims(img_array, 0)  # Create batch axis

#predictions = model.predict(img_array)
#score = float(predictions[0])
#print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")


