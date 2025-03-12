"""Simple forward pass to test the trained model."""

import json
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import torch
from sklearn.preprocessing import StandardScaler

from lstm import LSTMNet


def load_params(params_path: str) -> Dict[Any, Any]:
    """Load parameters from a JSON file.

    Args:
        params_path (str): The path to the JSON file containing the parameters.

    Returns:
        Dict[Any, Any]: A dictionary containing the loaded parameters.
    """
    with open(params_path, "r") as f:
        params: Dict[Any, Any] = json.load(f)
    return params


def load_model(
    model_path: str, input_size: int, hidden_size: int, lstm_hidden_size: int, output_size: int, nlayers: int, device: str
) -> LSTMNet:
    """Load a pre-trained LSTM model from a file.

    Args:
        model_path (str): The path to the saved model file.
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden layer.
        lstm_hidden_size (int): The size of the LSTM hidden layer.
        output_size (int): The size of the output layer.
        nlayers (int): The number of LSTM layers.
        device (str): The device to load the model onto (e.g., 'cpu' or 'cuda').

    Returns:
        LSTMNet: The loaded LSTM model.
    """
    model = LSTMNet(input_size, hidden_size, lstm_hidden_size, output_size, nlayers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_scaler(scaler_params: Dict[str, List[float]]) -> StandardScaler:
    """Load a StandardScaler with the given parameters.

    Args:
        scaler_params (Dict[str, List[float]]): A dictionary containing the mean and scale parameters for the scaler.

    Returns:
        StandardScaler: A StandardScaler object initialized with the given parameters.
    """
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params["mean"])
    scaler.scale_ = np.array(scaler_params["scale"])
    return scaler


def predict(model: LSTMNet, data: npt.NDArray[np.float64], scaler: StandardScaler, device: str) -> npt.NDArray[np.float64]:
    """Predict the output for the given data using the trained LSTM model.

    Args:
        model (LSTMNet): The trained LSTM model.
        data (npt.NDArray[np.float64]): The input data to predict.
        scaler (StandardScaler): The scaler used to preprocess the data.
        device (str): The device to run the prediction on (e.g., 'cpu' or 'cuda').

    Returns:
        npt.NDArray[np.float64]: The predicted output.
    """
    data_scaled = scaler.transform(data)

    X = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        y_pred = model(X)
        predicted = torch.sigmoid(y_pred).cpu().numpy()

    return predicted


if __name__ == "__main__":

    model_path = "checkpoints/best_model.pt"
    params_path = "checkpoints/params.json"

    params = load_params(params_path)

    scaler_params = params["scaler"]
    model_params = params["model"]

    input_size = model_params["input_size"]
    hidden_size = model_params["hidden_size"]
    lstm_hidden_size = model_params["lstm_hidden_size"]
    output_size = model_params["output_size"]
    nlayers = model_params["nlayers"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        model_path,
        input_size,
        hidden_size,
        lstm_hidden_size,
        output_size,
        nlayers,
        device,
    )
    scaler = load_scaler(scaler_params)

    new_data = np.array(
        [
            [
                -56.100006103515625,
                -136.62001037597656,
                254.6807861328125,
                -50.46002197265625,
                -125.1300048828125,
                57.5960693359375,
                268.739990234375,
                -122.07000732421875,
                269.152587890625,
                -175.98001098632812,
                187.52999877929688,
                93.02490234375,
                -101.79000854492188,
                158.73001098632812,
                44.77001953125,
            ]
        ]
    )

    predictions = predict(model, new_data, scaler, device)
    print("Result :", predictions)

    assert np.isclose(predictions[0], 1.0, atol=1e-3)
