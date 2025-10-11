import re
import json
import argparse

def format_chat_template(messages):
    chat_string = ""
    for message in messages:
        chat_string += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return chat_string

def parse_line_chat(input_file, output_file, my_name):
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
        f_out.write(json.dumps({"text": format_chat_template(messages)}, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse LINE chat logs.")
    parser.add_argument("--input_file", required=True, help="Path to the input LINE chat log file.")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file.")
    parser.add_argument("--my_name", required=True, help="Your name in the chat log.")
    args = parser.parse_args()

    parse_line_chat(args.input_file, args.output_file, args.my_name)