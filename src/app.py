from flask import Flask, request, jsonify
import torch
import data_processing as dp
from game_adjustment_model import train_model, adjust_game_parameters
from content_generation import load_llama_model, generate_content, generate_voice_content
from src.voice_module import transcribe_audio, capture_voice_input

app = Flask(__name__)

# Load and preprocess data
normalized_metrics_df = dp.normalize_metrics(dp.player_metrics_df)
high_interest_areas = dp.generate_heatmap()
X_tensor = torch.FloatTensor(normalized_metrics_df.values)
Y_tensor = torch.FloatTensor([[0.1, -0.05], [0.2, -0.1], [-0.1, 0.05], [0.0, 0.0], [0.15, -0.05]])

# Train the model
model = train_model(X_tensor, Y_tensor, X_tensor.shape[1], Y_tensor.shape[1])
scaler = dp.scaler  # Use the same scaler from data_processing

# Load LLaMA model for content generation
llama_model, llama_tokenizer, device = load_llama_model()


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
    mission = generate_content(llama_model, llama_tokenizer, device, area_description)
    return jsonify({"generated_mission": mission})

# Route to capture voice input and generate NPC interaction
@app.route('/voice_interaction', methods=['GET'])
def voice_interaction():
    # Capture voice input
    audio = capture_voice_input()
    # Transcribe audio to text
    transcribed_text = transcribe_audio(audio)
    if transcribed_text:
        # Use the transcribed text to generate NPC response
        npc_response = generate_npc_response(transcribed_text)
        return jsonify({"npc_response": npc_response})
    else:
        return jsonify({"error": "Could not understand audio"}), 400


def generate_npc_response(player_input):
    # Use the content generation module to create NPC dialogue
    prompt = f"Player says: '{player_input}'\nNPC responds:"
    npc_response = generate_voice_content(llama_model, llama_tokenizer, device, prompt)
    return npc_response


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
