from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model_direct(model_path="./mps-finetuned"):
    """Load model and tokenizer directly without pipeline"""
    print("Loading your fine-tuned model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,  # Handle positional embedding size mismatch
            torch_dtype=torch.float32,
        )

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("✅ Model loaded successfully!")
        print(f"📊 Model vocab size: {model.config.vocab_size}")
        print(f"📊 Positional embeddings: {model.transformer.wpe.weight.shape}")
        return model, tokenizer

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None


def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text using the model directly"""
    try:
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
            )

        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new part
        new_text = full_response[len(prompt) :].strip()

        return new_text if new_text else "[Model generated empty response]"

    except Exception as e:
        return f"Error generating response: {e}"


def simple_chat(model, tokenizer):
    """Simple chat loop"""
    print("\n" + "=" * 50)
    print("🤖 SIMPLE CHAT WITH YOUR MODEL")
    print("=" * 50)
    print("Type your message and press Enter")
    print("Type 'quit' to stop")
    print("-" * 50)

    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()

            # Check for exit
            if user_input.lower() in ["quit", "exit", "bye", "q"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            # Generate and print response
            print("🤖 Bot: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break


def test_model(model, tokenizer):
    """Test the model with a few prompts"""
    print("\n🧪 Testing model with simple prompts:")
    print("-" * 40)

    test_prompts = [
        "The weather today is",
        "Artificial intelligence is",
        "Wikipedia contains",
        "Hello, how are",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        response = generate_response(model, tokenizer, prompt, max_new_tokens=20)
        print(f"Response: '{response}'")


def main():
    print("🚀 Starting Simple Model Chat Test")

    # Load the model
    model, tokenizer = load_model_direct()

    if model is None:
        print("Failed to load model. Make sure './mps-finetuned' exists!")
        return

    # Test the model first
    test_model(model, tokenizer)

    # Start chat
    simple_chat(model, tokenizer)


if __name__ == "__main__":
    main()
