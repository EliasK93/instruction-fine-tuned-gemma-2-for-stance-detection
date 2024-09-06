import collections
import pandas
import transformers
import torch
from sklearn.metrics import classification_report
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from tools import clean_response, build_prompt


if __name__ == '__main__':

    # configurations
    TEMPERATURE = 0.0
    BATCH_SIZE = 8

    for MODEL_ID in ["google/gemma-2-9b-it", "trained_models/gemma2-9b-it-stance-finetuned/checkpoint-2000"]:

        # configure and load either base or fine-tuned model and tokenizer
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, attn_implementation='eager')
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, add_eos_token=True)

        # initialize a huggingface text generation pipeline for inference
        pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)

        # load data
        df = pandas.read_excel("data/test.xlsx", index_col=0)

        # format test set prompts
        prompts = []
        for topic, text in zip(df["topic"], df["text"]):
            prompts.append([{"role": "user", "content": build_prompt(topic, text)}])

        # run chunks of prompts through pipeline to get predictions
        predictions = []
        for response in pipeline(prompts, max_new_tokens=256, temperature=TEMPERATURE, batch_size=BATCH_SIZE):
            predictions.append(clean_response(response[0]["generated_text"][-1]['content']))
        df["prediction"] = predictions

        # calculate evaluation metrics
        report = classification_report(y_true=df["stance"], y_pred=df["prediction"])
        print(MODEL_ID)
        print(report)
        print("\nclass frequencies:", dict(collections.Counter(df["prediction"]).most_common()))
