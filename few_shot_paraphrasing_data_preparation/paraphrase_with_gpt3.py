import pandas as pd
import sys
import openai

openai.api_key = "YOUR API KEY"

task_name = sys.argv[0]

prompts_data = pd.read_csv(f"prompt_{task_name}.csv")
paraphrased_data = []

for index, row in prompts_data.iterrows():
    prompt = row['prompt']

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=64,
        n=1,
        stop=None,
        temperature=0.7,
    )

    generated_text = response["choices"][0]["text"]
    paraphrased_data.append(generated_text)

res = pd.DataFrame({"text": prompts_data["text"].to_list(), "paraphrased": paraphrased_data})
res.to_csv(f"../paraphrased_data/paraphrased_gpt3_{task_name}.csv", index=False)