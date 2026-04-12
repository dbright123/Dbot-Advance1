
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data_scaled, seq_len, pred_steps = 5, y_location = -1):
    """
    data_scaled: np.array shaped (n_rows, n_features)
    returns X (n_samples, seq_len, n_features), y (n_samples, n_features)
    where y is the row immediately following the window.
    """
    X, y = [], []
    n_rows = data_scaled.shape[0]
    for i in range(n_rows - seq_len - pred_steps + 1):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[1 + i+seq_len : 1 + i + seq_len+pred_steps, y_location])
    X = np.array(X)
    y = np.array(y)
    return X, y




def preprocess_and_save_scalers(data_x, data_y, scaler_x_filename='scaler_x.joblib', scaler_y_filename='scaler_y.joblib'):
    """
    Preprocesses 3D X and 2D Y data, scales them to a range of [0, 1],
    and saves the fitted scalers.

    Args:
        data_x (np.ndarray): The input features, a 3D numpy array of shape (samples, timesteps, features).
        data_y (np.ndarray): The target values, a 2D numpy array of shape (samples, output_dim).
        scaler_x_filename (str): Filename to save the scaler for X data.
        scaler_y_filename (str): Filename to save the scaler for Y data.

    Returns:
        tuple: A tuple containing the scaled X data (3D) and scaled Y data (2D).
    """
    try:
        # --- Preprocess X data (3D) ---
        # Reshape 3D data to 2D for scaler
        original_shape_x = data_x.shape
        data_x_reshaped = data_x.reshape(-1, original_shape_x[2])

        # Initialize and fit the scaler for X
        scaler_x = MinMaxScaler(feature_range=(0, 1))
        scaled_x_reshaped = scaler_x.fit_transform(data_x_reshaped)

        # Reshape the scaled data back to its original 3D shape
        scaled_x = scaled_x_reshaped.reshape(original_shape_x)

        # --- Preprocess Y data (2D) ---
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaled_y = scaler_y.fit_transform(data_y)

        # --- Save the scalers ---
        joblib.dump(scaler_x, scaler_x_filename)
        joblib.dump(scaler_y, scaler_y_filename)
        print(f"Scaler for X saved to '{scaler_x_filename}'")
        print(f"Scaler for Y saved to '{scaler_y_filename}'")

        return scaled_x, scaled_y

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None, None

def transform_data(data_x=None, data_y=None, scaler_x_filename='scaler_x.joblib', scaler_y_filename='scaler_y.joblib'):
    """
    Transforms new 3D X and/or 2D Y data using saved scalers.

    Args:
        data_x (np.ndarray, optional): The new input features (3D numpy array). Defaults to None.
        data_y (np.ndarray, optional): The new target values (2D numpy array). Defaults to None.
        scaler_x_filename (str): Filename of the saved scaler for X data.
        scaler_y_filename (str): Filename of the saved scaler for Y data.

    Returns:
        tuple: A tuple containing (scaled_x, scaled_y). Elements will be None if not provided.
    """
    scaled_x = None
    scaled_y = None

    # Transform X data if provided
    if data_x is not None:
        if not os.path.exists(scaler_x_filename):
            print(f"Error: Scaler file for X not found at '{scaler_x_filename}'")
        else:
            try:
                scaler_x = joblib.load(scaler_x_filename)
                original_shape_x = data_x.shape
                # Ensure there's data to reshape
                if original_shape_x[0] > 0:
                    data_x_reshaped = data_x.reshape(-1, original_shape_x[2])
                    scaled_x_reshaped = scaler_x.transform(data_x_reshaped)
                    scaled_x = scaled_x_reshaped.reshape(original_shape_x)
                else:
                    # Handle empty array case
                    scaled_x = np.array([]).reshape(original_shape_x)
            except Exception as e:
                print(f"An error occurred during X transformation: {e}")

    # Transform Y data if provided
    if data_y is not None:
        if not os.path.exists(scaler_y_filename):
            print(f"Error: Scaler file for Y not found at '{scaler_y_filename}'")
        else:
            try:
                scaler_y = joblib.load(scaler_y_filename)
                scaled_y = scaler_y.transform(data_y)
            except Exception as e:
                print(f"An error occurred during Y transformation: {e}")

    return scaled_x, scaled_y


def inverse_transform_data(scaled_x=None, scaled_y=None, scaler_x_filename='scaler_x.joblib', scaler_y_filename='scaler_y.joblib'):
    """
    Inverse transforms scaled 3D X and/or 2D Y data to their original scale.

    Args:
        scaled_x (np.ndarray, optional): The scaled input features (3D numpy array). Defaults to None.
        scaled_y (np.ndarray, optional): The scaled target values or predictions (2D numpy array). Defaults to None.
        scaler_x_filename (str): Filename of the saved scaler for X data.
        scaler_y_filename (str): Filename of the saved scaler for Y data.

    Returns:
        tuple: A tuple containing (inversed_x, inversed_y). Elements will be None if not provided.
    """
    inversed_x = None
    inversed_y = None

    # Inverse transform X data if provided
    if scaled_x is not None:
        if not os.path.exists(scaler_x_filename):
            print(f"Error: Scaler file for X not found at '{scaler_x_filename}'")
        else:
            try:
                scaler_x = joblib.load(scaler_x_filename)
                original_shape_x = scaled_x.shape
                # Ensure there's data to reshape
                if original_shape_x[0] > 0:
                    scaled_x_reshaped = scaled_x.reshape(-1, original_shape_x[2])
                    inversed_x_reshaped = scaler_x.inverse_transform(scaled_x_reshaped)
                    inversed_x = inversed_x_reshaped.reshape(original_shape_x)
                else:
                    inversed_x = np.array([]).reshape(original_shape_x)
            except Exception as e:
                print(f"An error occurred during X inverse transformation: {e}")

    # Inverse transform Y data if provided
    if scaled_y is not None:
        if not os.path.exists(scaler_y_filename):
            print(f"Error: Scaler file for Y not found at '{scaler_y_filename}'")
        else:
            try:
                scaler_y = joblib.load(scaler_y_filename)
                inversed_y = scaler_y.inverse_transform(scaled_y)
            except Exception as e:
                print(f"An error occurred during Y inverse transformation: {e}")

    return inversed_x, inversed_y
