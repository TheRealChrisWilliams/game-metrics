# game_adjustment_model.py

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# Define the neural network model
class GameParameterAdjustmentNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(GameParameterAdjustmentNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
            nn.Tanh()  # Outputs between -1 and 1
        )

    def forward(self, x):
        return self.model(x)


# Dataset class
class PlayerMetricsDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Training the model
def train_model(X_tensor, Y_tensor, input_size, output_size, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GameParameterAdjustmentNN(input_size, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Prepare data loader
    dataset = PlayerMetricsDataset(X_tensor, Y_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        for batch_X, batch_Y in dataloader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model


# Predict game adjustments
def adjust_game_parameters(model, scaler, new_metrics):
    new_metrics_normalized = scaler.transform([new_metrics])
    new_metrics_tensor = torch.FloatTensor(new_metrics_normalized)
    with torch.no_grad():
        adjustments = model(new_metrics_tensor)
    return adjustments[0].tolist()  # difficulty_adjustment, health_bar_adjustment
