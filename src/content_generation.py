import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI()

quadrant_names = {
    'Q1': 'Mystic Mountains',
    'Q2': 'Enchanted Forest',
    'Q3': 'Crystal Caverns',
    'Q4': 'Sunken Ruins'
}


# Generate content for high-interest areas
def generate_content(quadrant, player_metrics):
    quadrant_name = quadrant_names.get(quadrant, "the unknown lands")
    performance = player_metrics.get('performance', 'average')
    engagement = player_metrics.get('engagement', 'neutral')

    # Adjust the prompt based on player metrics
    if performance == 'high':
        challenge = "Create a challenging mission"
    elif performance == 'low':
        challenge = "Create an easier mission"
    else:
        challenge = "Create an engaging mission"

    if engagement == 'high':
        theme = "that expands the lore of"
    elif engagement == 'low':
        theme = "that revitalizes interest in"
    else:
        theme = "set in"

    prompt = f"{challenge} {theme} {quadrant_name}. The mission should match the game's style and captivate the player."

    try:
        response = openai.chat.completions.create(
            model='gpt-4o',
            messages=[
                {'role': 'system', 'content': 'You are a game content generator assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=4096,
            temperature=0.7,
        )
        mission_description = response.choices[0].message.content
    except Exception as e:
        print(f"Error generating content: {e}")
        mission_description = "An error occurred while generating content."

    return mission_description


# Generate NPC responses based on voice input
def generate_voice_content(prompt):
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': 'You are an NPC in a game responding to the player.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        npc_response = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating NPC response: {e}")
        npc_response = "An error occurred while generating NPC response."

    return npc_response
