import os
import sys
import json
from datetime import datetime
from google import genai
from google.genai import types

def main():
    # Set Google API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it using: export GOOGLE_API_KEY=your_key_here")
        sys.exit(1)
    
    # Configure the Gemini API
    client=genai.Client(api_key=api_key)
    
    # Get model configuration path from environment variable
    config_path = os.getenv("GEMINI_MODEL_CONFIG")
    print(config_path)
    if not config_path or not os.path.exists(config_path):
        print("Error: GEMINI_MODEL_CONFIG environment variable not set or file not found.")
        config_path = input("Enter the path to your Gemini model configuration file: ")
        if not os.path.exists(config_path):
            print(f"Error: File not found: {config_path}")
            sys.exit(1)
    
    # Load the model configuration
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
        print(f"Using model configuration from: {config_path}")
        
        # Extract examples for few-shot learning
        examples = config["examples"]
        print(f"Loaded {len(examples)} examples for few-shot learning")
        
        # Extract parameters
        model_name = config.get("model", "models/gemini-2.0-flash")
        temperature = config["parameters"].get("temperature", 0.2)
        top_p = config["parameters"].get("top_p", 0.95)
        top_k = config["parameters"].get("top_k", 40)
        
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        sys.exit(1)
    
    # Define test prompts in Hinglish - vary complexity and domains
    test_prompts = [
        "Aaj weather kaisa hai?",
        "Mujhe ek chai pilao.",
        "Weekend plan kya hai tumhara?",
        "Mera phone battery low ho raha hai, kya karoon?",
        "Movie recommendation do, thriller type ki",
        "Train ka time kya hai Delhi to Mumbai?",
        "Alexa, lights on kardo bedroom mein",
        "Calculator open karke 245 aur 378 add karo",
        "Kya tum mujhse Hindi-English mixed language mein baat kar sakte ho?",
        "Office ke liye kaunsa route best rahega aaj rush hour mein?"
    ]
    
    # Create output directory for saving results
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Current timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_dir}/inference_results_{timestamp}.json"
    
    # Store all results
    all_results = []
    
    
    print(f"\nTesting Gemini Flash 2.0 with {len(test_prompts)} Hinglish prompts...\n")
    print("=" * 60)
    
    # Generate responses for each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/{len(test_prompts)}")
        print(f"Prompt: {prompt}")
        
        try:
            # Create few-shot prompt with examples
            # Use only 3 most relevant examples to keep prompt shorter
            few_shot_prompt = "You are a helpful assistant that responds in Hinglish, a mix of Hindi and English.\n\nExamples:\n"
            
            # Add examples as few-shot demonstrations
            # In a production system, you might want to use retrieval to find the most relevant examples
            sample_examples = examples[:3]  # Using first 3 for simplicity
            for example in sample_examples:
                few_shot_prompt += f"User: {example['input_text']}\nAssistant: {example['output_text']}\n\n"
            
            # Add the current query
            few_shot_prompt += f"User: {prompt}\nAssistant:"
            
            # Generate response
            response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            temperature= temperature,
            top_p= top_p,
            top_k= top_k,
        ),
        contents=few_shot_prompt,
    )
    
            response_text = response.text.strip()
            
            print(f"Response: {response_text}")
            print("-" * 60)
            
            # Store result
            result = {
                "prompt": prompt,
                "response": response_text,
                "temperature": temperature,
                "few_shot_examples": len(sample_examples)
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            print("-" * 60)
    
    # Save all results to a JSON file
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAll test results saved to {results_file}")
    
    # Quick analysis of results
    print("\nQuick Analysis:")
    print(f"- Tested prompts: {len(test_prompts)}")
    print(f"- Successful responses: {len(all_results)}")
    print("- Check the output file for detailed results")
    

if __name__ == "__main__":
    main()