# merge_and_export.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # merge trên CPU để tránh OOM
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Merge LoRA weights vào base model
model = PeftModel.from_pretrained(base_model, "./gemma-legal-lora")
merged_model = model.merge_and_unload()

merged_model.save_pretrained("./gemma-legal-merged")
tokenizer.save_pretrained("./gemma-legal-merged")
print("✅ Merge hoàn tất!")