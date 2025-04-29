import os
import sys
import json
from datetime import datetime
import time
from google import genai
from google.genai import types
from google.api_core import exceptions

def main():
    # Set your Google API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it using: export GOOGLE_API_KEY=your_key_here")
        sys.exit(1)
    
    # Configure the Gemini API
    client=genai.Client(api_key=api_key)
    
    # Check if the dataset file exists
    dataset_path = "dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)
    
    print(f"Step 1: Preparing dataset file {dataset_path} for tuning...")
    
    # Check if the dataset file exists
    dataset_path = "dataset.jsonl"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)
    
    print(f"Step 1: Loading dataset file {dataset_path}...")
    
    # Load the dataset
    try:
        examples = []
        with open(dataset_path, "r", encoding="utf-8") as file:
            for line in file:
                example = json.loads(line)
                examples.append(example)
        
        print(f"✓ Loaded {len(examples)} examples from dataset")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    print("\nStep 2: Creating tuning job for Gemini Flash 2.0...")
    
    # Note: Since Gemini Flash 2.0 doesn't support the same fine-tuning API as OpenAI,
    # we'll create a prompt-based solution that uses the examples from our dataset
    
    # Save the model configuration with our tuning approach
    model_config = {
        "model": "models/gemini-2.0-flash",
        "tuning_type": "few-shot-examples",
        "examples": examples,
        "created_at": datetime.now().isoformat(),
        "parameters": {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40
        }
    }
    
    # Save configuration
    config_path = "gemini_model_config.json"
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(model_config, file, indent=2, ensure_ascii=False)
    
    print(f"✓ Tuning configuration created and saved to {config_path}")
    
    # Set environment variable for inference script
    os.environ["GEMINI_MODEL_CONFIG"] = config_path
    print(f"Environment variable GEMINI_MODEL_CONFIG has been set to: {config_path}")
    
    # For Windows
    if sys.platform == "win32":
        print("\nTo set the environment variable permanently on Windows:")
        print(f'setx GEMINI_MODEL_CONFIG "{config_path}"')
        print("Or set it in System Properties > Environment Variables.")
    # For Unix-like systems
    else:
        print("\nTo set the environment variable for future terminal sessions:")
        print(f'echo \'export GEMINI_MODEL_CONFIG="{config_path}"\' >> ~/.bashrc')
        print(f'echo \'export GEMINI_MODEL_CONFIG="{config_path}"\' >> ~/.zshrc  # If using zsh')
        print("Then restart your terminal or run: source ~/.bashrc")
    
    print("\nStep 3: Testing the model with few-shot learning...")
    
    # Initialize the model
    try:
        # Create a test prompt using our examples for few-shot learning
        test_prompt = "Kaisa feel kar rahe ho aaj kal?"
        few_shot_prompt = "You are a helpful assistant that responds in Hinglish, a mix of Hindi and English.\n\nExamples:\n"
        
        # Add 3 examples as few-shot demonstrations
        for i, example in enumerate(examples):
            few_shot_prompt += f"User: {example['input_text']}\nAssistant: {example['output_text']}\n\n"
        
        # Add the test query
        few_shot_prompt += f"User: {test_prompt}\nAssistant:"   


        print("The models : ",client.models.list())
        for model in client.models.list():
            print('\nl')
            print(model)
        print("The model : ",client.models.get(model="gemini-2.0-flash"))
        
        # Generate response
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                temperature= 0.2,
                top_p= 0.95,
                top_k= 40,
            ),
            contents = few_shot_prompt,
        )
        
        print("\nTest Result:")
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response.text}")
        
        print("\n✓ Test completed successfully!")
        print("✓ The inference.py script will use this few-shot learning approach")
        
    except Exception as e:
        print(f"Error testing model: {e}")
    
    print("\nNext steps:")
    print("1. Run the inference script: python inference.py")
    print("2. The inference script will use the examples from your dataset for few-shot learning")

if __name__ == "__main__":
    main()