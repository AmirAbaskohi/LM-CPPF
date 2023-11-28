import pandas as pd
import random
import sys
import copy

demonstration_template_number = int(sys.argv[1])
task_name = sys.argv[2]
instruction_template_number = None
if len(sys.argv) > 3:
    instruction_template_number = int(sys.argv[3])

demonstration_templates = []
instruction_templates = []

demonstration_template = "" 
instruction_template = ""

with open("demonstration_templates.txt", "r") as f:
    demonstration_templates = f.readlines()

demonstration_template = demonstration_templates[demonstration_template_number-1]

if instruction_template_number is not None:
    with open("instruction_templates.txt", "r") as f:
        instruction_templates = f.readlines()
    instruction_template = instruction_templates[instruction_template_number-1]

task_data = pd.read_csv(f"{task_name}.csv")
prompts_data = []

for index, row in task_data.iterrows():
    text = row["text"]
    label = row["label"]

    samples = task_data[task_data["label"] == label].sample(15)
    
    prompt = instruction_template
    prompt += "\n"

    for sample_index, sample_row in samples.iterrows():
        # Note: The following code is designed to collect user-inputted paraphrased samples due to the unavailability of shared data from QuillBot.
        # If deploying on a larger scale, consider automating this process, such as connecting to QuillBot or using an alternative model, to avoid manual input each time.
        # Please customize this section accordingly for your specific requirements.
        paraphrased = input(f"Paraphrase this: {sample_row['text']}")
        demonstration_template_copy = copy.copy(demonstration_template)
        demonstration_template_copy.replace("<Original>", sample_row['text'])
        demonstration_template_copy.replace("<Paraphrased>", paraphrased)

        prompt += demonstration_template_copy
        prompt += "\n"
    

    demonstration_template_copy = copy.copy(demonstration_template)
    demonstration_template_copy.replace("<Original>", text)
    demonstration_template_copy.replace("<Paraphrased>", "")
    prompt += demonstration_template_copy

    prompts_data.append(prompt)

res = pd.DataFrame({"text": task_data["text"].to_list(), "prompt": prompts_data})
res.to_csv(f"prompt_{task_name}.csv", index=False)
