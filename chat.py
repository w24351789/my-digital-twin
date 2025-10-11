import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    # Model and adapter paths
    model_name = "Qwen/Qwen1.5-7B-Chat"
    adapter_path = "/content/drive/MyDrive/my-digital-twin/naomi-chatbot-adapter"

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Load and merge the LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    print("Chatbot ready! Type 'quit' to exit.")

    # Chat loop
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Add user input to history
        history.append({"role": "user", "content": user_input})

        # Format the conversation history into ChatML format
        messages = tokenizer.apply_chat_template(
            history, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Tokenize the input
        model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")

        # Generate a response
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Add model response to history
        history.append({"role": "assistant", "content": response})

        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
