from flask import Flask, request, jsonify, render_template, url_for
import torch
import src.data_processing as dp
from src.game_adjustment_model import train_model, adjust_game_parameters
from src.content_generation import generate_content, generate_voice_content
from src.voice_module import transcribe_audio_whisper, record_audio
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variables to store data
mission = ""
npc_response = ""
heatmap_filename = "static/heatmap.png"

normalized_metrics_df = dp.normalize_metrics(dp.player_metrics_df)
max_quadrant = dp.generate_heatmap()
player_metrics_classification = dp.classify_player_metrics(normalized_metrics_df)
X_tensor = torch.FloatTensor(normalized_metrics_df.values)
Y_tensor = torch.FloatTensor([[0.1, -0.05], [0.2, -0.1], [-0.1, 0.05], [0.0, 0.0], [0.15, -0.05]])

# Train the model
model = train_model(X_tensor, Y_tensor, X_tensor.shape[1], Y_tensor.shape[1])
scaler = dp.scaler  # Use the same scaler from data_processing


@app.route('/')
def index():
    return render_template("index.html", mission=mission, npc_response=npc_response,
                           heatmap_url=url_for('static', filename='heatmap.png'),
                           player_metrics_classification=player_metrics_classification)


# Route to adjust game parameters
@app.route('/adjust_parameters', methods=['POST'])
def adjust_parameters():
    data = request.json
    new_metrics = data.get("metrics", [320, 0.75, 0.8, 0.65, 40])
    adjustments = adjust_game_parameters(model, scaler, new_metrics)
    return jsonify({"difficulty_adjustment": adjustments[0], "health_bar_adjustment": adjustments[1]})


# Route to generate content
@app.route('/generate_content', methods=['POST'])
def content():
    data = request.json
    area_description = data.get("area_description", "a mysterious forest")
    mission = generate_content(area_description, player_metrics_classification)
    return jsonify({"generated_mission": mission})


# Route to handle voice interaction
@app.route('/voice_interaction', methods=['GET'])
def voice_interaction():
    # Capture voice input
    audio = record_audio(duration=5, fs=16000)
    # Transcribe audio to text using Whisper
    transcribed_text = transcribe_audio_whisper(audio)
    if transcribed_text:
        # Generate NPC response
        npc_response = generate_voice_content(transcribed_text)
        return jsonify({"npc_response": npc_response})
    else:
        return jsonify({"error": "Could not transcribe audio"}), 400


def generate_npc_response(player_input):
    # Use the content generation module to create NPC dialogue
    prompt = f"Player says: '{player_input}'\nNPC responds:"
    npc_response = generate_voice_content(prompt)
    return npc_response


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
