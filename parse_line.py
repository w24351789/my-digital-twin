import re
import json
import argparse

def format_chat_template(messages):
    chat_string = ""
    for message in messages:
        chat_string += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return chat_string

def parse_line_chat(input_file, output_file, my_name, context_window=10):
    """
    Parses a LINE chat log file and converts it into a JSONL file
    with the ChatML format.
    """
    messages = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            # Regex to match the date line
            date_match = re.match(r'^\d{4}\.\d{2}\.\d{2} 星期.', line)
            if date_match:
                continue

            # Regex to match the timestamp, sender, and message
            chat_match = re.match(r'^(\d{2}:\d{2})\s(.*?)\s(.*)', line)
            if chat_match:
                _, sender, message_text = chat_match.groups()
                
                role = "assistant" if sender == my_name else "user"
                
                # If the role is the same as the last message, append the content
                if messages and messages[-1]["role"] == role:
                    messages[-1]["content"] += "\n" + message_text
                else:
                    messages.append({"role": role, "content": message_text})

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, message in enumerate(messages):
            if message["role"] == "assistant":
                # Create a training example with a sliding window of context
                start_index = max(0, i - context_window)
                history = messages[start_index:i+1]
                formatted_chat = format_chat_template(history)
                f_out.write(json.dumps({"text": formatted_chat}, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse LINE chat logs.")
    parser.add_argument("--input_file", required=True, help="Path to the input LINE chat log file.")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--my_name", required=True, help="Your name in the chat log.")
    parser.add_argument("--context_window", type=int, default=10, help="The number of previous messages to include as context.")
    args = parser.parse_args()

    parse_line_chat(args.input_file, args.output_file, args.my_name, args.context_window)