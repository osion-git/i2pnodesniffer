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
    y_true = df['prediction'].values
    X = df.drop(columns=['payload_bytes', 'timestamp', 'frame_num', 'old_frame_num', 'prediction', 'src_port',
                         'dst_port']).values
    return X, y_true


def main():
    csv_input = '../cnn.csv'
    df = pd.read_csv('../cnn.csv')
    print(df['prediction'].value_counts())
    X, y_true = load_and_preprocess(csv_input)
    # Scale
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    # Load modell
    model_path = 'saved_model/cnn_model_gross_3_ohne_port.keras'
    model = load_model(model_path)
    # Check dimension of model and input data, if no match, problem with prediction
    expected_input_shape = model.input_shape
    if X_cnn.shape[1:] != expected_input_shape[1:]:
        raise ValueError(f"Incorrect input dimensions: Expected {expected_input_shape[1:]}, but got {X_cnn.shape[1:]}.")
    # Prediction
    y_prob = model.predict(X_cnn, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    # Count prediction for each classes
    unique, counts = np.unique(y_pred, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class distribution on the entered data:")
    for cls, cnt in class_counts.items():
        print(f"Class {cls}: {cnt} samples")

    total = len(y_true)
    correct = int((y_pred == y_true).sum())
    accuracy = correct / total * 100
    print(f"\nCorrect predictions: {correct}/{total} ({accuracy:.2f}%)")

    # Each class accuracy
    classes = np.unique(y_true)
    for cls in classes:
        mask_cls = (y_true == cls)
        total_cls = mask_cls.sum()
        correct_cls = int(((y_pred == y_true) & mask_cls).sum())
        acc_cls = correct_cls / total_cls * 100
        print(f"Class {cls}: {correct_cls}/{total_cls} correct ({acc_cls:.2f}%)")


if __name__ == "__main__":
    main()
