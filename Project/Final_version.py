# Import necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# File paths for CSV and images
csv_path = "nutritional_values.csv"  # Path to CSV containing nutritional values
images_folder = "Dates/"  # Folder containing date fruit images

# Load the CSV into a DataFrame
df = pd.read_csv(csv_path)

# Ensure the CSV has all required columns
required_columns = ['image_id', 'calories', 'proteins', 'total_fat', 'glucose', 'cholesterol', 'water', 'Energy (Kcal)']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")

# Initialize lists to store image data (X) and nutritional values (y)
X = []
y = []
missing_images = 0  # Counter for missing image files

# Load and preprocess the images
for _, row in df.iterrows():
    # Construct the image file path
    image_path = os.path.join(images_folder, row['image_id'] + '.jpg')

    if os.path.exists(image_path):  # Check if the image file exists
        # Load and resize the image to 128x128, normalize pixel values to [0, 1]
        image = load_img(image_path, target_size=(128, 128))
        image = img_to_array(image) / 255.0
        X.append(image)  # Add the image data to the X list

        # Extract nutritional values from the CSV row
        y.append(row[['calories', 'proteins', 'total_fat', 'glucose', 'cholesterol', 'water', 'Energy (Kcal)']].values)
    else:
        missing_images += 1  # Increment missing image counter

# Log the number of missing images
print(f"Missing images: {missing_images}")

# Convert the lists into NumPy arrays
X = np.array(X)
y = np.array(y)

# Normalize the target values (nutritional values) using MinMaxScaler
scaler = MinMaxScaler()
y = scaler.fit_transform(y)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Convolutional layer
    MaxPooling2D((2, 2)),  # Max pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer
    MaxPooling2D((2, 2)),  # Max pooling layer
    Conv2D(128, (3, 3), activation='relu'),  # Convolutional layer
    MaxPooling2D((2, 2)),  # Max pooling layer
    Flatten(),  # Flattening layer
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout for regularization
    Dense(7, activation='linear')  # Output layer for 7 regression targets
])

# Compile the model with Adam optimizer and Mean Squared Error loss
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model with training and validation data
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,  # Number of training epochs
    batch_size=32  # Number of samples per batch
)

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Data augmentation for training data
datagen = ImageDataGenerator(
    rotation_range=30,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True  # Randomly flip images horizontally
)
datagen.fit(X_train)  # Fit the data generator to training data

# Predict the nutritional values on test data
predictions = model.predict(X_test)

# Inverse transform the predictions and true values to original scale
y_test_original = scaler.inverse_transform(y_test)
predictions_original = scaler.inverse_transform(predictions)


# Visualize true vs predicted values for a few test samples
def visualize_predictions(X, y_true, y_pred, num_samples=3):
    for i in range(num_samples):
        plt.figure(figsize=(12, 5))

        # Display the input image
        plt.subplot(1, 2, 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title("Input Image")

        # Display true vs predicted nutritional values
        plt.subplot(1, 2, 2)
        indices = range(len(y_true[i]))
        plt.plot(indices, y_true[i], label='True', color='blue', marker='o')
        plt.plot(indices, y_pred[i], label='Predicted', color='orange', marker='x')
        plt.xticks(indices, ['Calories', 'Proteins', 'Fat', 'Glucose', 'Cholesterol', 'Water', 'Energy'])
        plt.legend()
        plt.title("True vs Predicted Values")
        plt.show()


visualize_predictions(X_test, y_test_original, predictions_original)

# Save the trained model
model.save('001_nutritional_value_predictor.h5')