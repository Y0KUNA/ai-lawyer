# finetune_gemma_legal.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch


ds = load_dataset("undertheseanlp/UTS_VLC")
# --- Config ---
MODEL_ID = "google/gemma-2-2b-it"  # hoặc gemma-2-9b-it nếu đủ VRAM
OUTPUT_DIR = "./gemma-legal-lora"

# QLoRA: Load model 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# LoRA config — target attention layers
lora_config = LoraConfig(
    r=16,                    # rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → Chỉ train ~1-2% parameters, tiết kiệm VRAM

# --- Format prompt theo Gemma chat template ---
def format_sample(sample):
    prompt = f"""<start_of_turn>system
Bạn là chuyên gia tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi pháp lý DỰA TRÊN và CHỈ DỰA TRÊN các điều luật được cung cấp trong ngữ cảnh. 

Nguyên tắc bắt buộc:
1. Luôn trích dẫn rõ số điều, khoản, luật cụ thể
2. Nếu điều luật cung cấp không đủ căn cứ → nói rõ "Không đủ căn cứ pháp lý để trả lời"
3. Phân biệt rõ luật còn hiệu lực và đã hết hiệu lực
4. Không suy đoán ngoài phạm vi điều luật được cung cấp<end_of_turn>
<start_of_turn>user
[CÁC ĐIỀU LUẬT LIÊN QUAN]
{sample['context']}

[CÂU HỎI]
{sample['input']}<end_of_turn>
<start_of_turn>model
{sample['output']}<end_of_turn>"""
    return {"text": prompt}

# Load & format dataset
dataset = load_dataset("json", data_files={
    "train": "data/legal_train.jsonl",
    "validation": "data/legal_val.jsonl"
})
dataset = dataset.map(format_sample)

# --- Training ---
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    report_to="tensorboard",
    dataset_text_field="text",
    max_seq_length=4096,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)