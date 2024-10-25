from voice_module import record_audio, transcribe_audio_whisper
from content_generation import generate_voice_content
from content_generation import generate_content
from game_adjustment_model import adjust_game_parameters, train_model
import time
import torch
import data_processing as dp

player_metrics = {
    'completion_time': 0,
    'success_rate': 0.0,
    'accuracy': 0.0,
    'resource_utilization': 0.0,
    'session_duration': 0.0
}

start_time = time.time()

normalized_metrics_df = dp.normalize_metrics(dp.player_metrics_df)
max_quadrant = dp.generate_heatmap()
player_metrics_classification = dp.classify_player_metrics(normalized_metrics_df)
X_tensor = torch.FloatTensor(normalized_metrics_df.values)
Y_tensor = torch.FloatTensor([[0.1, -0.05], [0.2, -0.1], [-0.1, 0.05], [0.0, 0.0], [0.15, -0.05]])

# Train the model
model = train_model(X_tensor, Y_tensor, X_tensor.shape[1], Y_tensor.shape[1])
scaler = dp.scaler  # Use the same scaler from data_processing


def update_game_parameters():
    # Simulate new metrics (replace with actual collected data)
    new_metrics = [300, 0.8, 0.75, 0.6, 60]
    difficulty_adj, health_bar_adj = adjust_game_parameters(model, scaler, new_metrics)
    print(f"\n[Game Adjustment]")
    print(f"Difficulty Adjustment: {difficulty_adj:.2f}")
    print(f"Health Bar Adjustment: {health_bar_adj:.2f}")
    # Apply adjustments (simulate)
    if difficulty_adj > 0:
        print("Game difficulty has increased.")
    elif difficulty_adj < 0:
        print("Game difficulty has decreased.")
    else:
        print("Game difficulty remains the same.")


def present_personalized_mission(area_description):
    mission = generate_content(area_description, player_metrics_classification)
    print(f"\n[New Mission Generated]")
    print(f"Mission: {mission}")
    # Simulate player accepting the mission
    accept = input("Do you accept the mission? (yes/no): ")
    if accept.lower() == 'yes':
        print("Mission accepted!")
    else:
        print("Mission declined.")


def npc_interaction():
    print("\n[You approach an NPC]")
    print("NPC: 'Greetings traveler! What brings you here?'")
    # Record player's voice input
    audio = record_audio(duration=5, fs=16000)
    player_input = transcribe_audio_whisper(audio)
    if player_input:
        print(f"You said: '{player_input}'")
        # Generate NPC response
        npc_response = generate_voice_content(player_input)
        print(f"NPC: '{npc_response}'")
    else:
        print("NPC: 'I'm sorry, I didn't catch that.'")


def main_demo():
    print("Welcome to 'Echoes of the Enchanted Forest'!")
    time.sleep(1)
    print("You find yourself at the edge of a mysterious forest...")
    time.sleep(1)

    # Start tracking session duration
    session_start = time.time()

    # Simulate exploration
    print("\n[Exploring the Enchanted Forest]")
    time.sleep(2)
    # Update player metrics (simulated)
    player_metrics['completion_time'] = 300
    player_metrics['success_rate'] = 0.85
    player_metrics['accuracy'] = 0.8
    player_metrics['resource_utilization'] = 0.65
    player_metrics['session_duration'] = int(time.time() - session_start)

    # Update game parameters based on metrics
    update_game_parameters()

    # Generate personalized mission
    area_description = "the heart of the Enchanted Forest where ancient trees whisper secrets"
    present_personalized_mission(area_description)

    # Voice interaction with NPC
    npc_interaction()

    # Conclude the demo
    print("\nThank you for playing 'Echoes of the Enchanted Forest'!")


if __name__ == '__main__':
    main_demo()
