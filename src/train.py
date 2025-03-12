import json
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from lstm import LSTMNet


def evaluate(model: LSTMNet, data_loader: DataLoader[List[torch.Tensor]], device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device).float()
            y_pred = model(x)
            predicted = (torch.sigmoid(y_pred) > 0.5).float()
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total


def train(
    train_loader: DataLoader[List[torch.Tensor]],
    test_loader: DataLoader[List[torch.Tensor]],
    model: LSTMNet,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epochs: int = 100,
    checkpoint_path: str = "checkpoints/best_model.pt",
) -> None:
    max_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        print(f"Epoch {epoch}, train_loss: {loss.item()}")
        accuracy = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}, test_accuracy: {accuracy:.4f}")

        # Sauvegarder le modèle si l'accuracy est meilleure
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with accuracy {accuracy:.4f}")


def augment_data(
    features: npt.NDArray[np.float64], labels: npt.NDArray[np.bool], augmentation_factor: int = 2
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool]]:

    augmented_features = []
    augmented_labels = []

    for i in range(len(features)):
        for _ in range(augmentation_factor):
            # Jittering
            jittered = features[i] + np.random.normal(0, 0.01, features[i].shape)
            augmented_features.append(jittered)
            augmented_labels.append(labels[i])

            # Scaling
            scaled = features[i] * np.random.uniform(0.9, 1.1)
            augmented_features.append(scaled)
            augmented_labels.append(labels[i])

    return np.array(augmented_features), np.array(augmented_labels)


if __name__ == "__main__":
    dataset = load_dataset("pollen-robotics/anyskin_slip_detection", data_dir="data", token=True)

    features = np.array(
        [
            dataset["train"]["mag1_x"],
            dataset["train"]["mag1_y"],
            dataset["train"]["mag1_z"],
            dataset["train"]["mag2_x"],
            dataset["train"]["mag2_y"],
            dataset["train"]["mag2_z"],
            dataset["train"]["mag3_x"],
            dataset["train"]["mag3_y"],
            dataset["train"]["mag3_z"],
            dataset["train"]["mag4_x"],
            dataset["train"]["mag4_y"],
            dataset["train"]["mag4_z"],
            dataset["train"]["mag5_x"],
            dataset["train"]["mag5_y"],
            dataset["train"]["mag5_z"],
        ]
    ).T

    target = np.array(dataset["train"]["slip"])

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Exporter les paramètres du scaler
    scaler_params = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}

    augmented_features, augmented_labels = augment_data(features, target)

    # Convertir en tenseurs PyTorch
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32).view(-1, 1)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)

    # Créer des DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[2]
    hidden_size = 128
    lstm_hidden_size = 128
    output_size = 1
    nlayers = 1
    epochs = 100

    params = {
        "scaler": scaler_params,
        "model": {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "lstm_hidden_size": lstm_hidden_size,
            "output_size": output_size,
            "nlayers": nlayers,
            "epochs": epochs,
            "batch_size": batch_size,
        },
    }

    with open("checkpoints/params.json", "w") as f:
        json.dump(params, f)

    print(type(train_loader))

    print(type(test_loader))
    exit(0)

    device = "cuda"

    model = LSTMNet(input_size, hidden_size, lstm_hidden_size, output_size, nlayers)
    print(model)

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(train_loader, test_loader, model, criterion, optimizer, device, epochs)
