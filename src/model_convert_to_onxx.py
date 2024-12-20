import json
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle

def export_model_to_onnx(model, X_rating_train):
    """
    Exports a trained scikit-learn model to ONNX format and saves it as a file.

    Parameters:
    model (sklearn.base.BaseEstimator): The trained scikit-learn model to be exported.
    X_rating_train (numpy.ndarray): The training data used to determine the input shape for the model.

    The function converts the model into the ONNX format and serializes it to a file named "random_forest_model.onnx".
    """
    # Define the input type for the ONNX model (a tensor with the same number of features as X_rating_train)
    initial_type = [('float_input', FloatTensorType([None, X_rating_train.shape[1]]))]
    
    # Convert the scikit-learn model to ONNX format
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model to a file
    with open("random_forest_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

def save_scalers_as_json(scaler, scaler_y):
    """
    Saves the properties of two scalers (for features and target variable) as JSON files.

    Parameters:
    scaler (sklearn.preprocessing.StandardScaler): The scaler used for the features (X).
    scaler_y (sklearn.preprocessing.StandardScaler): The scaler used for the target variable (y).

    The function serializes the mean and scale properties of both scalers and saves them to JSON files:
    "scaler_x.json" for the feature scaler and "scaler_y.json" for the target variable scaler.
    """
    # Convert the scaler properties (mean and scale) into a dictionary format suitable for JSON serialization
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    scaler_y_data = {
        "mean": scaler_y.mean_.tolist(),
        "scale": scaler_y.scale_.tolist()
    }

    # Save the scaler data for X to a JSON file
    with open("scaler_x.json", "w") as f:
        json.dump(scaler_data, f)
    
    # Save the scaler data for y to a JSON file
    with open("scaler_y.json", "w") as f:
        json.dump(scaler_y_data, f)
