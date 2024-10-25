import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


# Load the LLaMA model
def load_llama_model(model_name='"meta-llama/Llama-3.2-1B'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
    model.to(device)
    return model, tokenizer, device


# Generate content for high-interest areas
def generate_content(model, tokenizer, device, area_description):
    prompt = f"Create a new mission where the player explores {area_description}."
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    mission_description = generated_text[len(prompt):].strip()
    return mission_description


def generate_voice_content(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the generated text after the prompt
    response = generated_text[len(prompt):].strip()
    return response
