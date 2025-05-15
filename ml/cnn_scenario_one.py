import ast
import os

import numpy as np
import pandas as pd
from keras.api import regularizers
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard

from helper.ml import undersample_df
from helper.preprocessing_cnn_helper import PreprocessingCnnHelper

np.random.seed(42)
# Read data
df = pd.read_csv('../cnn.csv')
df_corr = pd.read_csv('../cnn.csv')
df['payload_bytes'] = df['payload_bytes'].apply(ast.literal_eval)
df = PreprocessingCnnHelper.split_payload_bytes(df, payload_col='payload_bytes', prefix='byte', fill_value=0)
# Print correlation
print(df_corr.corr(numeric_only=True)['prediction'].sort_values(ascending=False))
# Take data to check
num_manual_samples_per_class = 100
df_0 = df[df['prediction'] == 0].sample(n=num_manual_samples_per_class)
df_1 = df[df['prediction'] == 1].sample(n=num_manual_samples_per_class)
df_manual_test = pd.concat([df_0, df_1])
df = df.drop(df_manual_test.index)
# Prevent a dominant class
df = undersample_df(df, target_col='prediction')
print(f'Einträge: {len(df)}')
# Split Dataset to x and y
y = df.pop('prediction').values
X = df.drop(columns=['payload_bytes', 'timestamp', 'frame_num', 'old_frame_num']).values
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42, stratify=y
)

# Build model
model = models.Sequential([
    layers.Conv1D(
        filters=64,
        kernel_size=2,
        activation=None,
        input_shape=(X_train.shape[1], 1)
    ),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(
        filters=32,
        kernel_size=3,
        activation=None,
        kernel_regularizer=regularizers.l2(0.008)
    ),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.GlobalMaxPooling1D(),
    layers.Dense(
        units=64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.008)
    ),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
log_dir = os.path.join("logs", "cnn")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=1e-4,
    verbose=1
)
#Training
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=16,
    validation_split=0.1,
    callbacks=[tensorboard_callback, reduce_lr, early_stopping]
)

#Save model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/cnn_model.keras")
# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

def predict_packet(df_single):
    x = df_single.values
    x_scaled = scaler.transform(x)
    x_cnn = x_scaled.reshape((1, x_scaled.shape[1], 1))
    prob = model.predict(x_cnn)[0, 0]
    return 1 if prob > 0.5 else 0
correct = 0
total = len(df_manual_test)

for i in range(total):
    df_sample = \
        df_manual_test.drop(columns=['payload_bytes', 'timestamp', 'frame_num', 'old_frame_num', 'prediction']).iloc[[i]]
    true_label = df_manual_test['prediction'].iloc[i]
    pred_label = predict_packet(df_sample)
    print(f"{i + 1}: True label = {true_label}, Predicted = {pred_label}")
    if pred_label == true_label:
        correct += 1
accuracy_percent = (100 * correct) / total
print(f"Test-Accuracy: {accuracy_percent:.2f}% ({correct}/{total} korrekt)")