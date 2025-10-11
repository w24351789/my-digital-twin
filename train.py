import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def train(args):
    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 2. Load dataset
    dataset = load_dataset('json', data_files=args.dataset_path, split='train')

    # 3. Set up LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"], # Adjust based on model architecture
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 4. Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        fp16=True, # Use fp16 for mixed-precision training
    )

    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        peft_config=lora_config,
        args=training_args,
        max_seq_length=1024,
    )

    # 6. Start training
    trainer.train()

    # 7. Save the LoRA adapter
    model.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen1.5-7B-Chat", help="The name of the base model to use.")
    parser.add_argument("--dataset_path", type=str, default="./data/training_dataset.jsonl", help="Path to the training dataset.")
    parser.add_argument("--output_dir", type=str, default="./naomi-chatbot-adapter", help="Directory to save the LoRA adapter.")
    args = parser.parse_args()
    train(args)