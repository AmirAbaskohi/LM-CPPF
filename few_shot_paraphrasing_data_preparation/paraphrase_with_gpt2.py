import pandas as pd
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2-medium"  # or "gpt2-medium", "gpt2-large", "gpt2-xl" for larger models
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

task_name = sys.argv[0]

prompts_data = pd.read_csv(f"prompt_{task_name}.csv")
paraphrased_data = []

for index, row in prompts_data.iterrows():
    prompt = row['prompt']

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    output = model.generate(input_ids, max_length=64, num_return_sequences=1, temperature=0.7)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    paraphrased_data.append(generated_text)

res = pd.DataFrame({"text": prompts_data["text"].to_list(), "paraphrased": paraphrased_data})
res.to_csv(f"../paraphrased_data/paraphrased_gpt2_{task_name}.csv", index=False)
