#!/usr/bin/env python3
import ast

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from helper.preprocessing_cnn_helper import PreprocessingCnnHelper


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    # If the payload is to be ignored, this and the next line must be commented out.
    df['payload_bytes'] = df['payload_bytes'].apply(ast.literal_eval)
    df = PreprocessingCnnHelper.split_payload_bytes(
        df,
        payload_col='payload_bytes',
        prefix='byte',
        fill_value=0
    )
    X = df.drop(columns=['payload_bytes', 'timestamp', 'frame_num', 'old_frame_num', 'prediction']).values
    return X


def main():
    csv_input = '../cnn_docker.csv'
    X = load_and_preprocess(csv_input)
    # Scale
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    # Load modell
    model_path = 'saved_model/cnn_model_grosser_datensatz_nur_payload.keras'
    model = load_model(model_path)
    # Check dimension of model and input data, if no match, problem with prediction
    expected_input_shape = model.input_shape
    if X_cnn.shape[1:] != expected_input_shape[1:]:
        raise ValueError(f"Incorrect input dimensions: Expected {expected_input_shape[1:]}, but got {X_cnn.shape[1:]}.")
    # Prediction
    y_prob = model.predict(X_cnn, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    # Count prediction for each classes
    unique, counts = np.unique(y_pred, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class distribution on the entered data:")
    for cls, cnt in class_counts.items():
        print(f"Class {cls}: {cnt} samples")


if __name__ == "__main__":
    main()
