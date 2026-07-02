import os
import sys
import json
import torch
import gc
import random
import numpy as np
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from torch.utils.data import SequentialSampler

# Custom Trainer to save/load optimizer state using safe-optim-checkpoint
class SafeSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, *args, **kwargs):
        return SequentialSampler(self.train_dataset)

    def _save_optimizer_and_scheduler(self, output_dir):
        opt_state = self.optimizer.state_dict()
        try:
            from optimtensors.serde import safe_save_optimizer
            safe_save_optimizer(opt_state, os.path.join(output_dir, "optimizer.safetensors"))
            print(f"--> Saved optimizer state using safe_save_optimizer to optimizer.safetensors")
        except Exception as e:
            print(f"--> Failed to save using safe_save_optimizer (expected for quantized): {e}")
            raise e
        # Save a dummy optimizer.pt so Hugging Face Trainer doesn't fail
        torch.save({}, os.path.join(output_dir, "optimizer.pt"))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            
    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        
        safetensors_path = os.path.join(checkpoint, "optimizer.safetensors")
        if os.path.exists(safetensors_path):
            print(f"--> Loading optimizer state from: {safetensors_path} using safe_load_into_optimizer")
            from optimtensors.serde import safe_load_into_optimizer
            safe_load_into_optimizer(self.optimizer, safetensors_path)
        else:
            print(f"--> safetensors file not found, falling back to pickle")
            super()._load_optimizer_and_scheduler(checkpoint)
            
        scheduler_path = os.path.join(checkpoint, "scheduler.pt")
        if os.path.exists(scheduler_path):
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path))


class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_step=15):
        self.stop_step = stop_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_save = True
            control.should_training_stop = True


def seed_everything(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_mock_dataset():
    data = [
        {"text": "Instruction: What is Botmartz?\nResponse: Botmartz is an advanced AI-powered course and developer document assistant designed to help developers learn system integration and agentic logic."},
        {"text": "Instruction: How do I load an optimizer checkpoint safely?\nResponse: You can use the safe-optim-checkpoint library's safe_load_optimizer function to map the binary buffers into the PyTorch optimizer without code execution."},
        {"text": "Instruction: What is the benefit of memory mapping checkpoints?\nResponse: Memory mapping enables O(1) RAM allocation, mapping file bytes directly without reading them into physical memory until they are read."},
        {"text": "Instruction: How do I run scale tests for safe-optim-checkpoint?\nResponse: Run the scale benchmarks using python run_scale_tests.py to trace BERT-Base and GPT-2 models."},
        {"text": "Instruction: Does safe-optim-checkpoint support mixed precision?\nResponse: Yes, it preserves FP16 and FP32 moment tensors exactly, reinterpreting BF16 safely through INT16 re-views."},
    ] * 20
    return Dataset.from_list(data)


def load_model_and_tokenizer(max_seq_length):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    print("Loading Llama-3.2-1B in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
    tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,  # Set to 0 to avoid dropout noise difference
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def main():
    max_seq_length = 512
    dataset = get_mock_dataset()
    
    # --- PHASE 1: Control Run (No Checkpoints) ---
    print("\n--- Running Control Run (adamw_torch) ---")
    seed_everything(3407)
    model, tokenizer = load_model_and_tokenizer(max_seq_length)
    
    training_args_control = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=30,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        output_dir="outputs_control",
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_length,
        dataset_num_proc=1,
        eos_token=tokenizer.eos_token,
        seed=3407,
    )
    
    trainer_control = SafeSFTTrainer(
        model=model,
        args=training_args_control,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    trainer_control.train()
    control_losses = [log["loss"] for log in trainer_control.state.log_history if "loss" in log]
    
    with open("losses_control.json", "w") as f:
        json.dump(control_losses, f)
    print(f"Control losses saved: {control_losses[:5]}... -> total {len(control_losses)} steps")
    
    # Free memory
    del trainer_control, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- PHASE 2: Interrupted & Resumed Run ---
    print("\n--- Running Interrupted Run (adamw_torch) - First Half (15 steps) ---")
    seed_everything(3407)
    model, tokenizer = load_model_and_tokenizer(max_seq_length)
    
    training_args_interrupted = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=30,  # Keep it at 30 so the learning rate schedule is identical
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="no",  # We handle save in our callback
        optim="adamw_torch",
        output_dir="outputs_interrupted",
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_length,
        dataset_num_proc=1,
        eos_token=tokenizer.eos_token,
        seed=3407,
    )
    
    trainer_interrupted = SafeSFTTrainer(
        model=model,
        args=training_args_interrupted,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[StopAtStepCallback(15)],
    )
    
    trainer_interrupted.train()
    first_half_losses = [log["loss"] for log in trainer_interrupted.state.log_history if "loss" in log]
    
    # Free first trainer but keep model for resuming
    del trainer_interrupted
    gc.collect()
    
    print("\n--- Resuming Run (adamw_torch) - Second Half (steps 15 to 30) ---")
    training_args_resume = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=30,  # Goal: step 30
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_torch",
        output_dir="outputs_interrupted",
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_length,
        dataset_num_proc=1,
        eos_token=tokenizer.eos_token,
        seed=3407,
    )
    
    trainer_resume = SafeSFTTrainer(
        model=model,
        args=training_args_resume,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    trainer_resume.train(resume_from_checkpoint="outputs_interrupted/checkpoint-15")
    
    second_half_losses = [log["loss"] for log in trainer_resume.state.log_history if "loss" in log]
    second_half_losses_filtered = []
    for log in trainer_resume.state.log_history:
        if "loss" in log and log["step"] > 15:
            second_half_losses_filtered.append(log["loss"])
            
    resumed_losses = first_half_losses + second_half_losses_filtered
    with open("losses_resumed.json", "w") as f:
        json.dump(resumed_losses, f)
        
    print(f"Resumed losses saved: {resumed_losses[:5]}... -> total {len(resumed_losses)} steps")
    
    print(f"Control Loss at step 15: {control_losses[14]:.4f}, step 16: {control_losses[15]:.4f}")
    print(f"Resumed Loss at step 15: {resumed_losses[14]:.4f}, step 16: {resumed_losses[15]:.4f}")
    
    for idx, (l_c, l_r) in enumerate(zip(control_losses, resumed_losses)):
        diff = abs(l_c - l_r)
        # Verify perfect matching (allow minor fp noise <= 0.01)
        assert diff < 0.01, f"Loss mismatch at step {idx+1}: Control={l_c:.4f}, Resumed={l_r:.4f}"
    print("SUCCESS: Loss curves match perfectly across checkpoint interrupt and resume!")
    
    # Free memory
    del trainer_resume, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- PHASE 3: 8-bit Quantized Optimizer Rejection (adamw_8bit) ---
    print("\n--- Running adamw_8bit Rejection Check ---")
    seed_everything(3407)
    model, tokenizer = load_model_and_tokenizer(max_seq_length)
    
    training_args_8bit = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        optim="adamw_8bit",
        output_dir="outputs_8bit",
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_length,
        dataset_num_proc=1,
        eos_token=tokenizer.eos_token,
        seed=3407,
    )
    
    trainer_8bit = SafeSFTTrainer(
        model=model,
        args=training_args_8bit,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    try:
        trainer_8bit.train()
        print("ERROR: Did not raise ValueError for 8-bit quantized optimizer!")
        sys.exit(1)
    except ValueError as e:
        print(f"SUCCESS: Correctly raised expected ValueError for 8-bit quantized optimizer state dict: {e}")
        assert "Unsupported optimizer state shape: 8-bit quantized optimizers" in str(e)
        
    print("\n🎉 ALL FRAMEWORK RESUME & REJECTION TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
