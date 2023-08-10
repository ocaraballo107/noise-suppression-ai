import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Load and preprocess data
# X_train, y_train: Training data (noisy speech and clean speech)
# X_test, y_test: Test data (noisy speech and clean speech)
# After loaded and preprocessed data into X_data and y_data

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Create the CNN model
model = Sequential()
# Add layers as shown in the previous code snippet

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(None, None, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output for fully connected layers
model.add(Flatten())

# Fully connected layers with ReLU and BatchNormalization
model.add(Dense(1024))
model.add(ReLU())
model.add(BatchNormalization())

# Regression layer
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# After data loaded and preprocessed
X_train: Input data (noisy speech)
y_train: Target data (clean speech)

# Define the number of epochs and batch size
epochs = 20
batch_size = 32

# Initialize variables to track minimum loss and corresponding epoch
min_loss = float('inf')
min_loss_epoch = None

# Train the model
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Shuffle the training data
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Iterate over batches
    for batch_start in range(0, len(X_train_shuffled), batch_size):
        batch_end = min(batch_start + batch_size, len(X_train_shuffled))
        batch_X = X_train_shuffled[batch_start:batch_end]
        batch_y = y_train_shuffled[batch_start:batch_end]
        
        # Train the model on the current batch
        model.train_on_batch(batch_X, batch_y)
    

    # Evaluate the model's performance on the test set
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    
    # Check if the current loss is the minimum
    if loss < min_loss:
        min_loss = loss
        min_loss_epoch = epoch + 1  # Adding 1 to get the actual epoch number

print("Training finished.")
print(f"Minimum loss achieved at Epoch {min_loss_epoch}, with loss value: {min_loss:.4f}")

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
