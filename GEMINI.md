# GEMINI.md: My Digital Twin Project

## Project Overview

This project aims to create a personalized AI chatbot, a "digital twin," that learns and mimics the unique LINE chatting style of "羽含 Naomi". The core of the project is to fine-tune the `Qwen1.5-7B-Chat` language model on a personal chat history to create a bot that can generate responses that are stylistically and tonally similar to the user.

The project uses the following technologies:
*   **Base Model:** `Qwen1.5-7B-Chat`
*   **Fine-tuning Technique:** PEFT (Parameter-Efficient Fine-tuning) with LoRA (Low-Rank Adaptation)
*   **Frameworks & Libraries:** Python, PyTorch, Hugging Face (Transformers, Datasets, PEFT, TRL), and `regex`.

The project follows a four-stage workflow:
1.  **Data Processing:** Parsing LINE chat logs from a `.txt` file into a structured format.
2.  **Dataset Construction:** Converting the processed data into the ChatML format for training.
3.  **Model Fine-tuning:** Fine-tuning the base model using the prepared dataset.
4.  **Interactive Inference:** Chatting with the fine-tuned model.

## Building and Running

Here are the key commands for the project:

1.  **Install Dependencies:**
    ```bash
    pip install torch transformers datasets peft trl
    ```

2.  **Process Raw Chat Logs:**
    Place the exported LINE chat log (`line_chat.txt`) in a `data/` directory. Then run:
    ```bash
    python parse_line.py --input_file data/line_chat.txt --output_file data/training_dataset.jsonl --my_name "i"
    ```

3.  **Start Model Fine-tuning:**
    ```bash
    python train.py
    ```

4.  **Chat with Your AI Twin:**
    ```bash
    python chat.py
    ```

## Development Conventions

*   **Language:** Python 3.9+
*   **Core Libraries:** The project heavily relies on the Hugging Face ecosystem for model loading, data processing, and training.
*   **Data Processing:** Raw LINE chat logs are parsed using the `parse_line.py` script, which uses regular expressions to extract messages, timestamps, and senders. It's important to handle privacy by removing or anonymizing sensitive information during this step.
*   **Dataset Format:** The training data is structured in the ChatML format, with the user's messages assigned the `assistant` role and the other person's messages assigned the `user` role. The final dataset is a `.jsonl` file.
*   **Model Training:** The `train.py` script uses the `SFTTrainer` from the TRL library to simplify the fine-tuning process with LoRA. The resulting LoRA adapter weights are saved to a directory (e.g., `./chatbot-adapter`).
*   **Inference:** The `chat.py` script loads the base model and merges it with the trained LoRA weights for interactive chat sessions.
