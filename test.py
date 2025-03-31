import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from prompts import sft_prompt, sft_inst
from evaluate import evaluator, load

# Paths
save_path = './other_sft/checkpoint'  # Path to the saved finetuned model

# Quantization configuration (same as training)
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
device_map = {"": 0}

# Sample sentences for testing
# sample_sentences = [
#     "The player is in a forest with trees and a river nearby. The goal is to build a bridge.",
#     "The player has mined 10 stones and is near a cliff. The goal is to reach the top.",
#     "The player is low on health and surrounded by enemies. The goal is to survive.",
# ]
import json

samples = []

with open('data/test.jsonl') as f:
    lines = f.readlines()
    for line in lines:
        samples.append(json.loads(line))

sample_prompts = [
    sft_prompt.format(system_message="", scene=sample['scene'], goal=sample['goal'], actions=sample['actions'], feedback=sample['feedback'])
    for sample in samples
]

# Quantization config (consistent with training)
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load the saved model with quantization
model = AutoModelForCausalLM.from_pretrained(
    save_path,
    quantization_config=bnb_config,
    device_map=device_map
)
model.eval()  # Set model to evaluation mode

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Create a pipeline for text generation
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def test_bleu(sample_sentences):
    
    bleu = load("bleu")

    # predictions = ["The cat is on the mat"]
    # references = [["The cat sits on the mat"]]
    preds = [generator(sentence, max_new_tokens=100, do_sample=True, temperature=0.7) for sentence in sample_sentences]
    refs = [[sentence['feedback'] for sentence in sample_prompts]]

    results = bleu.compute(predictions=preds, references=refs)
    print(f"BLEU Score: {results['bleu']}")

def test_model(sample_sentences):
    # Test the model with sample sentences
    print("\n--- Testing Model with Sample Sentences ---")
    for sentence in [sample_sentences[-7], sample_sentences[-10]]:
        # print(f"\nInput: {sentence}")
        output = generator(sentence, max_new_tokens=100, do_sample=True, temperature=0.7)
        print("----------------------")
        print(f"***Output: {output[0]['generated_text'][len(sentence)+1:]}")

def test_bleu(sample_sentences):
    
    bleu = load("bleu")

    # predictions = ["The cat is on the mat"]
    # references = [["The cat sits on the mat"]]
    preds = [generator(sentence, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)[0]['generated_text'][len(sentence)+1:] for sentence in sample_sentences]
    refs = [[sentence['feedback']] for sentence in samples]

    # print(preds)
    # print("-----")
    print(refs)

    results = bleu.compute(predictions=preds, references=refs)
    print(f"BLEU Score: {results['bleu']}")

# def test_text2text(sample_sentences):
    


# Run the test
if __name__ == "__main__":
    test_model(sample_prompts)