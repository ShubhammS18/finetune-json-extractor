import json

def convert_dpo_file(input_path, output_path):
    converted = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            # Extract prompt messages
            prompt_messages = record['prompt']
            # Extract chosen and rejected assistant content
            chosen_content   = record['chosen'][0]['content']
            rejected_content = record['rejected'][0]['content']
            # Fireworks DPO format
            new_record = {
                "messages": prompt_messages + [
                    {"role": "assistant", "content": chosen_content}
                ],
                "non_preferred_output": [
                    {"role": "assistant", "content": rejected_content}
                ]}
            
            converted.append(new_record)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Converted {len(converted)} records -> {output_path}")

convert_dpo_file('data/dpo_train.jsonl', 'data/dpo_train_converted.jsonl')
convert_dpo_file('data/dpo_val.jsonl',   'data/dpo_val_converted.jsonl')
