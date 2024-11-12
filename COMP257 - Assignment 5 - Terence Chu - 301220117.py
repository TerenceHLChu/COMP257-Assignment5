# Student name: Terence Chu
# Student number: 301220117

from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.layers import Dense, Input
from keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load Olivetti faces dataset
olivetti = fetch_olivetti_faces()

print('Olivetti faces data shape', olivetti.data.shape) # 64 × 64
print('Olivetti faces target shape', olivetti.target.shape)

X = olivetti.data
y = olivetti.target

print('\nPixel values:\n', X)
print('Pixel maximum:', X.max())
print('Pixel minimum:', X.min())
print('Data is already normalized')

# Display the first 12 images of the dataset
plt.figure(figsize=(7,7))

for i in range(12):
    plt.subplot(3, 4, i+1) # 3 rows, 4 columns
    plt.imshow(X[i].reshape(64,64), cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.suptitle("First 12 images of the dataset", fontsize=16, y=0.9)
plt.show()

# Create an instance of PCA and preserve 99% of X's variance
pca = PCA(n_components=0.99)
X_reduced = pca.fit_transform(X)

print(f'\nShape of the reduced X dataset: {X_reduced.shape}') # PCA reduced data to 260 dimensions

# As noted above, the Olivetti dataset is normalized by default
# However, after PCA's dimensionality reduction, the pixel ranges are not longer between 0.0 and 1.0
# Use min max scaler to scale the pixels back to between 0.0 and 1.0
scaler = MinMaxScaler(feature_range=(0, 1))
X_reduced_scaled = scaler.fit_transform(X_reduced)

# Display the first 12 images of the dimension-reduced X dataset 
plt.figure(figsize=(7,7))

for i in range(12):
    plt.subplot(3, 4, i+1) # 3 rows, 4 columns
    
    # Reverse mix max scaler and PCA reduction to map the reduced data back to its original space
    X_reversed_scaler = scaler.inverse_transform(X_reduced_scaled)
    plt.imshow(pca.inverse_transform(X_reversed_scaler)[i].reshape(64,64), cmap='gray')
    
    plt.xticks([])
    plt.yticks([])

plt.suptitle("First 12 images of the reduced X dataset", fontsize=16, y=0.9)
plt.show()

# Split the dataset into train, test, and validation sets
# stratify=y ensures the class distribution in each split set is the same as the original dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(X_reduced_scaled, y, stratify=y, test_size=0.2, random_state=17)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, stratify=y_train_full, test_size=0.25, random_state=17)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('X_valid shape:', X_valid.shape)

def create_autoencoder(input_size, learning_rate, regularizer_rate):
    hidden_size = 128 # For the encoding and decoding layers
    code_size = 32 # For the latent space (bottleneck)
    
    input_img = Input(shape=(input_size,)) # Shape of the flattened image
    hidden1 = Dense(hidden_size, activation='relu', kernel_regularizer=regularizers.l2(regularizer_rate))(input_img)
    code = Dense(code_size, activation='relu', kernel_regularizer=regularizers.l2(regularizer_rate))(hidden1)
    hidden2 = Dense(hidden_size, activation='relu', kernel_regularizer=regularizers.l2(regularizer_rate))(code)
    output_img = Dense(input_size, activation='sigmoid')(hidden2)
    
    autoencoder = Model(input_img, output_img)
    
    optimizer = Adam(learning_rate=learning_rate)
    
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    return autoencoder

# Initialize k-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=17)

# Hyperparameter grid for tuning
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
regularizer_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

# Intialize the best validation loss (to hold the lowest validation loss)
best_validation_loss = 999

# Initialize dictionary to hold the best parameters
best_params = {}

# Loop through the hyperparameter grid
for learning_rate in learning_rates:
    for regularizer_rate in regularizer_rates:
        # To hold the validation loss of each fold
        k_fold_validation_losses = []

        # kf.split(X_reduced_scaled) produce k number of indices (k=5 as defined by n_splits above)
        # Four of the indices (80%) are assigned to train_index
        # The remaining index (20%) is assigned to val_index
        # This loop will execute 5 times, with the val_index swapping to another index each execution - EACH EXECUTION OF THE LOOP IS ONE FOLD
        # This loop implements k-fold cross validation
        for train_index, val_index in kf.split(X_reduced_scaled):
            X_train, X_val = X_reduced_scaled[train_index], X_reduced_scaled[val_index] # Split the input data (X_reduced_scaled) into train and validation sets
            
            # Create and fit an autoencoder with values from the hyperparameter grid with the training data
            model = create_autoencoder(input_size=X_train.shape[1], learning_rate=learning_rate, regularizer_rate=regularizer_rate)
            model.fit(X_train, X_train, epochs=5, verbose=0)
            
            # Calculate the loss on the validation set (based on the loss function specified when the model was compiled)
            # Append the each fold's loss to the list
            # After the loop has executed k (5) times, the list will contain the (5) loss values (one for each fold)
            validation_loss = model.evaluate(X_val, X_val, verbose=0)
            k_fold_validation_losses.append(validation_loss)
            
        # Calculate the mean validation loss for the current hyperparameter combination
        mean_validation_loss = np.mean(k_fold_validation_losses)
        print(f'Learning rate: {learning_rate} | Regularizer rate: {regularizer_rate} | Mean validation loss: {mean_validation_loss}')

        # Update the best parameters if the current loss is better (lower)
        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            best_params = {'learning_rate': learning_rate, 'regularizer_rate': regularizer_rate}
            
print("\nBest hyperparameters:")
print(best_params)

# Train a new model with the best hyperparameters
best_param_model = create_autoencoder(input_size=X_train.shape[1], learning_rate=best_params['learning_rate'], regularizer_rate=best_params['regularizer_rate'])
best_param_model.fit(X_train, X_train, epochs=5)

# Make prediction (reconstruction) on the test set
decoded_img = best_param_model.predict(X_test)

n = 10

plt.figure(figsize=(20, 4))
for i in range(n):
    # Original image
    ax = plt.subplot(2, n, i + 1)
    
    # Reverse mix max scaler and PCA reduction to map the reduced data back to its original space
    X_reversed_scaler = scaler.inverse_transform(X_test)
    original_img = pca.inverse_transform(X_reversed_scaler[i]).reshape(64, 64)
 
    plt.imshow(original_img, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    # Reconstructed image
    ax = plt.subplot(2, n, i + 1 + n)    
    X_reversed_scaler = scaler.inverse_transform(decoded_img)
    decoded_original_space = pca.inverse_transform(X_reversed_scaler[i]).reshape(64, 64)
    plt.imshow(decoded_original_space, cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.suptitle('Original images (top row) vs. reconstructed images (bottom row)', fontsize=16, y=0.95)
plt.show()