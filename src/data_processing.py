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

quadrant_names = {
    'Q1': 'Mystic Mountains',
    'Q2': 'Enchanted Forest',
    'Q3': 'Crystal Caverns',
    'Q4': 'Sunken Ruins'
}


def classify_player_metrics(normalized_metrics_df):
    # For simplicity, we'll use Success Rate and Session Duration
    last_session = normalized_metrics_df.iloc[-1]
    performance_metric = last_session['Success Rate']
    engagement_metric = last_session['Session Duration']

    # Classify performance
    if performance_metric > 0.8:
        performance = 'high'
    elif performance_metric < 0.6:
        performance = 'low'
    else:
        performance = 'average'

    # Classify engagement
    if engagement_metric > 0.7:
        engagement = 'high'
    elif engagement_metric < 0.3:
        engagement = 'low'
    else:
        engagement = 'average'

    return {'performance': performance, 'engagement': engagement}


def get_quadrant(position, world_size):
    x, y = position
    mid_x, mid_y = world_size[0] // 2, world_size[1] // 2
    if x < mid_x and y < mid_y:
        return 'Q3'  # Bottom-left
    elif x < mid_x and y >= mid_y:
        return 'Q1'  # Top-left
    elif x >= mid_x and y < mid_y:
        return 'Q4'  # Bottom-right
    else:
        return 'Q2'  # Top-right


# Normalize the metrics
def normalize_metrics(df):
    normalized_data = scaler.fit_transform(df)
    return pd.DataFrame(normalized_data, columns=df.columns)


# World Heatmap Generation
def generate_heatmap():
    world_size = (10, 10)
    output_filename = "heatmap.png"
    player_positions = [
        # Positions in Q1 (Top-left)
        (2, 6), (2, 6), (3, 7), (2, 8), (3, 6),
        (2, 6), (3, 7), (2, 7), (3, 6), (2, 6),
        (3, 7), (2, 8), (3, 6), (2, 6), (2, 6),
        (3, 7), (2, 8), (3, 6), (2, 6), (2, 6),
        (7, 2), (8, 3), (7, 2), (8, 3), (7, 2),
    ]

    heatmap_data = np.zeros(world_size)
    quadrant_activity = {'Q1': 0, 'Q2': 0, 'Q3': 0, 'Q4': 0}
    for position in player_positions:
        heatmap_data[position] += 1
        quadrant = get_quadrant(position, world_size)
        quadrant_activity[quadrant] += 1

    heatmap_normalized = heatmap_data / heatmap_data.max()
    plt.imshow(heatmap_normalized, cmap='hot', interpolation='nearest')
    plt.title('Player Activity Heatmap')
    plt.colorbar()
    # Save the heatmap to a file
    plt.savefig(output_filename)
    plt.close()

    # Identify the quadrant with the highest activity
    max_quadrant = max(quadrant_activity, key=quadrant_activity.get)
    print(f"Quadrant with highest activity: {quadrant_names[max_quadrant]} ({max_quadrant})")
    return max_quadrant


normalized_metrics_df = normalize_metrics(player_metrics_df)
max_quadrant = generate_heatmap()
player_metrics_classification = classify_player_metrics(normalized_metrics_df)
