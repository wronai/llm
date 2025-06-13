from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    print("Testing model loading...")
    
    # Use a small model for testing
    model_name = "microsoft/DialoGPT-small"
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32,  # Use float32 for CPU
        trust_remote_code=True
    )
    
    print("\nModel loaded successfully!")
    print(f"Device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test inference
    print("\nTesting inference...")
    prompt = "Witaj! Jak się masz?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nModel response:")
    print(response)

if __name__ == "__main__":
    main()
