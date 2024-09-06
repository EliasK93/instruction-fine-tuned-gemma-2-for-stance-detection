## Parameter-Efficient Fine-Tuning of Gemma-2-Instruct-9B for Stance Detection 

Example application for fine-tuning the instruction-tuned 9B variant of Google's [Gemma 2](https://blog.google/technology/developers/google-gemma-2/) model ([google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)) on a new task (Stance Detection). 

To make the fine-tuning feasible on a consumer GPU, a parameter-efficient fine-tuning (PEFT) approach based on [QLoRA](https://arxiv.org/abs/2305.14314) (_Quantized Low-Rank Adaptation_) is applied. While the original model has 9.295.724.032 parameters, this approach works with only 54.018.048 trainable parameters (~0.581%), leaving the original model weights entirely frozen during the fine-tuning process. This way, the model was fine-tuned for two epochs on the task using a single RTX 4080 GPU.

<br>

### Corpus

Each model was fine-tuned on a 5,000 sentences Stance Detection corpus that I manually annotated during my Master's Thesis.
Stance Detection aims to classify the stance a sentence takes towards a claim (topic) as either _Pro_, _Contra_ or _Neutral_.
The sentences originate from Reddit's _r/ChangeMyView_ subreddit in the time span between January 2013 and October 2018, as provided in the [ConvoKit subreddit corpus](https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/).
They cover five topics: _abortion_, _climate change_, _gun control_, _minimum wage_ and _veganism_.
The table below shows some examples.

<table>
<thead>
<tr>
<th align="center">topic</th>
<th align="center">sentence</th>
<th align="center">stance label</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">There should be more gun control.</td>
<td align="left">It's the only country with a "2nd Amendment", yet 132 countries have a lower murder rate.</td>
<td align="center">Pro</td>
</tr>
<tr>
<td align="center">Humanity needs to combat climate change.</td>
<td align="left">The overhwelming evidence could be lies and you would never know because you're content to live your life as a giant appeal to authority.</td>
<td align="center">Contra</td>
</tr>
<tr>
<td align="center">Vegans are right.</td>
<td align="left">It's all about finding a system that works for you.</td>
<td align="center">Neutral</td>
</tr>
</tbody></table>

<br>

For the instruction-based fine-tuning and inference, the sentence pairs are wrapped in the following prompt:

<table>
<thead>
<tr>
<th align="center">prompt</th>
<th align="center">expected output</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left">The first text is a hypothesis/claim, the second text is a sentence. Determine whether the sentence is a pro argument ("pro"), a contra argument ("con") or doesnt take position at all/is neutral ("neu") towards the hypothesis.
<br>For your answer, just write exactly one of pro, con or neu, not a full text.

<br>Sentence to predict:
<br>hypothesis: \`\`\`There should be more gun control.\`\`\`
<br>sentence: \`\`\`It's the only country with a "2nd Amendment", yet 132 countries have a lower murder rate.\`\`\`
<br>stance:</td>
<td align="center">pro</td>
</tr>
</tbody></table>

<br>

### Results

|                      Model                       | Accuracy | Micro-F1 | Macro-F1 |
|:------------------------------------------------:|:--------:|:--------:|:--------:|
|        base model (google/gemma-2-9b-it)         |   0.72   |   0.71   |   0.71   |
| fine-tuned model (gemma2-9b-it-stance-finetuned) |   0.90   |   0.89   |   0.90   |

<br>

### Requirements

##### - Python >= 3.10

##### - Conda
  - `pytorch==2.4.0`
  - `cudatoolkit=12.1`

##### - pip
  - `transformers`
  - `datasets`
  - `trl`
  - `sentencepiece`
  - `protobuf`
  - `peft`
  - `bitsandbytes`
  - `openpyxl`
  - `scikit-learn`

<br>

### Notes

The dataset files in this repository are cut off after the first 50 rows.
The trained model files `adapter_model.safetensors`, `optimizer.pt` and `tokenizer.json` are omitted in this repository.
