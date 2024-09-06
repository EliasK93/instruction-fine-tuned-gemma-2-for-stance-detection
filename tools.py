import re
import pandas
from datasets import Dataset
from transformers import PreTrainedTokenizer


def load_tokenized_dataset(path: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Load the dataset, and return it as a tokenized Dataset.
    """
    df = pandas.read_excel(path, index_col=0)
    df["prompt"] = [build_prompt(topic, text) for topic, text in zip(df["topic"], df["text"])]
    df["dialogue"] = [build_dialogue(prompt, stance) for prompt, stance in zip(df["prompt"], df["stance"])]
    dataset = Dataset.from_dict({col: df[col] for col in ["dialogue"]})
    return dataset.map(lambda samples: tokenizer(samples["dialogue"]), batched=True)


def build_prompt(topic: str, sentence: str) -> str:
    """
    Wrap topic hypothesis and potentially argumentative sentence into a prompt for Stance Detection.
    """
    return f"""The first text is a hypothesis/claim, the second text is a sentence. Determine whether the sentence is a pro argument ("pro"), a contra argument ("con") or doesnt take position at all/is neutral ("neu") towards the hypothesis.
For your answer, just write exactly one of pro, con or neu, not a full text.

Sentence to predict:
hypothesis: ```{topic}```
sentence: ```{sentence}```
stance:"""


def build_dialogue(prompt: str, output: str) -> str:
    """
    Wrap prompt and expected model output into the Gemma instruction chat template (https://huggingface.co/google/gemma-2-9b-it#chat-template).
    """
    return f"""<start_of_turn>user {prompt} <end_of_turn>\n<start_of_turn>model {output} <end_of_turn>"""


def clean_response(response: str) -> str:
    """
    Since Gemma does not always exactly follow the expected format and the instruction chat template, allow some very
    minor deviations in the answer to still count as a correct answer, specifically:
    - remove any trailing or leading whitespaces or linebreaks from the model response
    - remove leading 'model ' prefixes as the instruction model sometimes incorrectly includes it twice
    Map any still not parsable result (any response that is not 'pro', 'con' or 'neu') to 'invalid'.
    """
    response = response.strip()
    response = re.sub('^model\s+', '', response)
    response = response.strip()
    if response not in ["pro", "con", "neu"]:
        response = "invalid"
    return response
