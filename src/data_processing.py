import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Initialize scaler for normalization
scaler = MinMaxScaler()

# Sample player metrics data
player_metrics_data = [
    [300, 0.8, 0.75, 0.6, 60],
    [250, 0.9, 0.85, 0.7, 45],
    [400, 0.6, 0.65, 0.5, 30],
    [350, 0.7, 0.80, 0.65, 50],
    [275, 0.85, 0.90, 0.75, 55],
]
metrics_columns = ['Completion Time', 'Success Rate', 'Accuracy', 'Resource Utilization', 'Session Duration']
player_metrics_df = pd.DataFrame(player_metrics_data, columns=metrics_columns)


# Normalize the metrics
def normalize_metrics(df):
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns)


# World Heatmap Generation
def generate_heatmap():
    world_size = (10, 10)
    player_positions = [
        (2, 3), (2, 3), (2, 4), (3, 3), (2, 3),
        (7, 8), (7, 8), (7, 9), (8, 8), (7, 8),
        (2, 3), (2, 3), (3, 3), (2, 2), (2, 3),
        (5, 5), (5, 5), (5, 6), (6, 5), (5, 5),
        (2, 3), (2, 3), (2, 4), (3, 3), (2, 3),
    ]
    heatmap_data = np.zeros(world_size)
    for position in player_positions:
        heatmap_data[position] += 1

    # Normalize and visualize the heatmap
    heatmap_normalized = heatmap_data / heatmap_data.max()
    plt.imshow(heatmap_normalized, cmap='hot', interpolation='nearest')
    plt.title('Player Activity Heatmap')
    plt.colorbar()
    plt.show()

    # Identify high-interest areas
    high_interest_areas = np.argwhere(heatmap_normalized > 0.5)
    return high_interest_areas


# Load and preprocess data
normalized_metrics_df = normalize_metrics(player_metrics_df)
