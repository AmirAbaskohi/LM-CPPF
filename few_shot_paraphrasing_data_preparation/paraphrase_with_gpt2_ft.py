import pandas as pd
import sys
import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Set the path to the input data file
data_file = "para-nmt-50m-small.txt"

model_name = "gpt2-medium"  # or "gpt2-medium", "gpt2-large", "gpt2-xl" for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

with open(data_file, "r", encoding="utf-8") as file:
    lines = file.read().split("\n")

original_sentences = []
paraphrase_sentences = []
for line in lines:
    if line:
        original, paraphrase = line.split("\t")
        original_sentences.append(original.strip())
        paraphrase_sentences.append(paraphrase.strip())

# Fine-tune GPT-2 model or load the fine-tuned model if available
if os.path.exists("./fine-tuned-model"):
    fine_tuned_model = GPT2LMHeadModel.from_pretrained("./fine-tuned-model")
else:
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_file,
        block_size=128,
        overwrite_cache=True,
    )
    config = GPT2Config.from_pretrained(model_name, output_hidden_states=True)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir="./fine-tuned-model",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=1e-3,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    trainer.train()
    fine_tuned_model = trainer.model

fine_tuned_model.eval()

task_name = sys.argv[0]

prompts_data = pd.read_csv(f"prompt_{task_name}.csv")
paraphrased_data = []

for index, row in prompts_data.iterrows():
    prompt = row['prompt']

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = fine_tuned_model.generate(input_ids, max_length=64, num_return_sequences=1, temperature=0.7)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    paraphrased_data.append(generated_text)

res = pd.DataFrame({"text": prompts_data["text"].to_list(), "paraphrased": paraphrased_data})
res.to_csv(f"../paraphrased_data/paraphrased_fine_tuned_gpt2_{task_name}.csv", index=False)
