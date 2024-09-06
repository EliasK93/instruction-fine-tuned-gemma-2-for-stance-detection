import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import transformers
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import hf_token
import tools

"""
Note: Parts of the model configuration and data formatting are based on this blog:
https://huggingface.co/blog/gemma-peft
"""


if __name__ == "__main__":

    # configurations
    MODEL_ID = "google/gemma-2-9b-it"
    OUTPUT_DIR = "trained_models/gemma2-9b-it-stance-finetuned"
    LORA_LAYERS = ["o_proj", "v_proj", "q_proj", "o_proj"]
    LORA_ALPHA = 32
    LORA_R = 16
    LORA_DROPOUT = 0.05
    BATCH_SIZE = 4
    NUM_EPOCHS = 2

    # use 4-bit quantization, 4-bit NormalFloat and nested quantization and set computation dtype to BFloat16
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)
    # load the actual model and apply quantization configuration
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, token=hf_token.token,
                                                 attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, add_eos_token=True)

    # setup model for LoRA application
    model = prepare_model_for_kbit_training(model)
    # set dimension of the LoRA attention, LoRA alpha, LoRA dropout and linear layers to be targeted
    lora_config = LoraConfig(r=LORA_R, lora_dropout=LORA_DROPOUT, target_modules=LORA_LAYERS, lora_alpha=LORA_ALPHA,
                             task_type="CAUSAL_LM")
    # apply LoRA configuration
    model = get_peft_model(model, lora_config)
    # ensure pad_token and padding_side are properly set
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # load data
    dataset_train = tools.load_tokenized_dataset("data/train.xlsx", tokenizer=tokenizer)
    dataset_val = tools.load_tokenized_dataset("data/val.xlsx", tokenizer=tokenizer)

    # train model using the SFTTrainer from the Transformer Reinforcement Learning (trl) library
    args = transformers.TrainingArguments(output_dir=OUTPUT_DIR, learning_rate=0.0001, seed=1, weight_decay=0.01,
                                          per_device_train_batch_size=BATCH_SIZE, num_train_epochs=NUM_EPOCHS,
                                          per_device_eval_batch_size=BATCH_SIZE, eval_strategy="epoch",
                                          save_strategy="epoch", load_best_model_at_end=True)
    trainer = SFTTrainer(model=model, train_dataset=dataset_train, eval_dataset=dataset_val, peft_config=lora_config,
                         args=args, data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
                         dataset_text_field="dialogue", max_seq_length=1024)
    trainer.train()
