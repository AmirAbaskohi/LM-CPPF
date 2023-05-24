# Paraphrase Data Preparation

In order to run the experiments based on paraphrased data, you need to run these code. First you run the prompt_generator by giving information about the templates you want to use for paraphrasing and then it will generate a `.csv` for you that has the necessary prompts.

Then you can use these prompts to generate the paraphrased data. For this you have two options:

1- Using GPT-3: For this a code is avaialble for you that reads the input dataframe; however it needs a GPT-3 API Key.
2- Using OPT-175B: As there is no API endpoint avialable, you can write a scraper or run the prompt by going to official API in: [https://opt.alpa.ai/](https://opt.alpa.ai/)