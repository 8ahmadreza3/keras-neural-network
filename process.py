# Import required libraries
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split

# Set Keras backend to TensorFlow
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# 1. Load the datasets
train = pd.read_csv('./data/petfinder_train.csv')
test = pd.read_csv('./data/petfinder_test.csv')

# 2. Create the Target column
# Animals with AdoptionSpeed 0-3 are adopted (True), otherwise False
train['Target'] = np.where(train['AdoptionSpeed'] < 4, True, False)

# 3. Drop unnecessary columns
train.drop(columns=['AdoptionSpeed', 'Description'], inplace=True)
test.drop(columns=['Description'], inplace=True)

# 4. Encode ordinal variables (MaturitySize, FurLength, Health)
le = LabelEncoder()

# MaturitySize
train['MaturitySize'] = le.fit_transform(train['MaturitySize'])
test['MaturitySize'] = le.transform(test['MaturitySize'])

# FurLength
train['FurLength'] = le.fit_transform(train['FurLength'])
test['FurLength'] = le.transform(test['FurLength'])

# Health
train['Health'] = le.fit_transform(train['Health'])
test['Health'] = le.transform(test['Health'])

# 5. Encode nominal variables
# Type and Gender (only 2 categories, so use LabelEncoder)
le = LabelEncoder()
train['Type'] = le.fit_transform(train['Type'])
test['Type'] = le.transform(test['Type'])

train['Gender'] = le.fit_transform(train['Gender'])
test['Gender'] = le.transform(test['Gender'])

# Use BinaryEncoder for columns with more categories (Breed1, Color1, Color2, Vaccinated, Sterilized)
binary_encoder = ce.BinaryEncoder(cols=['Breed1', 'Color1', 'Color2', 'Vaccinated', 'Sterilized'])

# Apply BinaryEncoder to train and test
train_binary = binary_encoder.fit_transform(train[['Breed1', 'Color1', 'Color2', 'Vaccinated', 'Sterilized']])
test_binary = binary_encoder.transform(test[['Breed1', 'Color1', 'Color2', 'Vaccinated', 'Sterilized']])

# Concatenate binary-encoded columns and drop original columns
train = pd.concat([train, train_binary], axis=1)
test = pd.concat([test, test_binary], axis=1)
columns = ['Breed1', 'Color1', 'Color2', 'Vaccinated', 'Sterilized']
train.drop(columns=columns, inplace=True)
test.drop(columns=columns, inplace=True)

# 6. Normalize numerical and encoded columns
# Normalize all columns except Target using mean and std from train
columns = [col for col in train.columns if col != 'Target']
for column in columns:
    mean = train[column].mean()
    std = train[column].std()
    train[column] = (train[column] - mean) / std
    test[column] = (test[column] - mean) / std

# 7. Save preprocessed datasets
train.to_csv('train_preprocessed.csv', index=False)
test.to_csv('test_preprocessed.csv', index=False)

# 8. Split train data into training and validation sets (90% train, 10% validation)
X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop(columns=['Target']), train['Target'], test_size=0.1, random_state=42
)

# Print dataset sizes
print('Train examples:', len(X_train), len(y_train))
print('Validation examples:', len(X_valid), len(y_valid))
print('Test examples:', len(test))

# 9. Build the neural network
model = keras.Sequential()
model.add(keras.layers.Input(shape=(X_train.shape[1],)))  # Input layer with number of features
model.add(keras.layers.Dense(5000, activation='relu'))    # Dense layer with 5000 neurons, ReLU activation
model.add(keras.layers.Dense(1000, activation='relu'))    # Dense layer with 1000 neurons, ReLU activation
model.add(keras.layers.Dense(500, activation='relu'))     # Dense layer with 500 neurons, ReLU activation
model.add(keras.layers.Dense(1, activation='sigmoid'))    # Output layer with sigmoid activation for binary classification

# 10. Display model summary
model.summary()

# 11. Compile the model
model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# 12. Train the model
epochs = 10
BATCH_SIZE = 128

history = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=epochs,
                    validation_data=(X_valid, y_valid))

# 13. Predict on test data
predictions = model.predict(test)
submission = pd.DataFrame({'Target': (predictions > 0.5).flatten()})

# 14. Save predictions
submission.to_csv('submission.csv', index=False)

# 15. Combine test data with predictions and save
test_with_predictions = pd.concat([pd.read_csv('test_preprocessed.csv'), submission], axis=1)
test_with_predictions.to_csv('test_with_predictions.csv', index=False)

# 16. Display first few rows of saved files for verification
print("\nTrain Preprocessed Data (first 5 rows):")
print(pd.read_csv('train_preprocessed.csv').head())

print("\nTest Preprocessed Data (first 5 rows):")
print(pd.read_csv('test_preprocessed.csv').head())

print("\nSubmission Data (first 5 rows):")
print(pd.read_csv('submission.csv').head())

print("\nTest Data with Predictions (first 5 rows):")
print(pd.read_csv('test_with_predictions.csv').head())

# 17. Evaluate model on validation set using F1 score
from sklearn.metrics import f1_score
val_predictions = model.predict(X_valid)
val_predictions_binary = (val_predictions > 0.5).flatten()
f1 = f1_score(y_valid, val_predictions_binary, average='weighted')
print(f'\nF1 Score on validation set: {f1}')