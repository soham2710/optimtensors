import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler

try:
    from transformers import Trainer, TrainingArguments, TrainerCallback
    from optimtensors.integrations import (
        OptimTensorsCallback,
        OptimTensorsTrainerMixin,
        INTEGRATIONS_AVAILABLE,
    )
except ImportError:
    INTEGRATIONS_AVAILABLE = False


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(5, 2)
        
    def forward(self, x, labels=None):
        logits = self.classifier(x)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


class TinyDataset(Dataset):
    def __init__(self, size=80, seed=42):
        torch.manual_seed(seed)
        self.x = torch.randn(size, 5)
        self.labels = torch.randint(0, 2, (size,))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {"x": self.x[idx], "labels": self.labels[idx]}


def tiny_collate_fn(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class SafeTrainer(OptimTensorsTrainerMixin, Trainer):
    """Subclass Trainer using the Mixin for safe loading."""
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=tiny_collate_fn,
        )


class LossTrackerCallback(TrainerCallback):
    """Callback to record training step losses."""
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])


def get_reproducible_setup():
    """Initializes seeds, model weights, and datasets identically for reproduction."""
    torch.manual_seed(1337)
    import random
    random.seed(1337)
    import numpy as np
    np.random.seed(1337)
    
    dataset = TinyDataset(size=64)
    model = TinyModel()
    return model, dataset


@pytest.mark.skipif(not INTEGRATIONS_AVAILABLE, reason="Hugging Face Transformers not available")
def test_trainer_integration_dual_and_strict():
    # ----------------------------------------------------
    # 1. CONTROL RUN: Train 6 steps uninterrupted
    # ----------------------------------------------------
    model, dataset = get_reproducible_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        loss_tracker = LossTrackerCallback()
        
        training_args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=6,
            per_device_train_batch_size=8,
            logging_steps=1,
            seed=1337,
            data_seed=1337,
            full_determinism=True,
            report_to="none",
        )
        
        trainer = SafeTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[loss_tracker],
        )
        trainer.train()
        control_losses = list(loss_tracker.losses)
        assert len(control_losses) == 6

    # ----------------------------------------------------
    # 2. DUAL MODE: Interrupt at step 3, resume from check
    # ----------------------------------------------------
    model, dataset = get_reproducible_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        loss_tracker = LossTrackerCallback()
        opt_callback = OptimTensorsCallback(mode="dual")
        
        training_args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=3,
            per_device_train_batch_size=8,
            logging_steps=1,
            save_steps=3,
            seed=1337,
            data_seed=1337,
            full_determinism=True,
            report_to="none",
        )
        
        trainer = SafeTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[loss_tracker, opt_callback],
        )
        trainer.train()
        interrupted_losses = list(loss_tracker.losses)
        
        checkpoint_dir = os.path.join(tmpdir, "checkpoint-3")
        assert os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt"))
        assert os.path.isfile(os.path.join(checkpoint_dir, "optimizer.optimtensors"))
        
        # Resume training from step 3 using SafeTrainer
        loss_tracker_resume = LossTrackerCallback()
        training_args_resume = TrainingArguments(
            output_dir=tmpdir,
            max_steps=6,
            per_device_train_batch_size=8,
            logging_steps=1,
            seed=1337,
            data_seed=1337,
            full_determinism=True,
            report_to="none",
        )
        
        trainer_resume = SafeTrainer(
            model=model,
            args=training_args_resume,
            train_dataset=dataset,
            callbacks=[loss_tracker_resume],
        )
        trainer_resume.train(resume_from_checkpoint=checkpoint_dir)
        resumed_losses = list(loss_tracker_resume.losses)
        
        # Validate loss curve matches control to 3 decimal places
        combined_losses = interrupted_losses + resumed_losses
        for i in range(6):
            assert round(combined_losses[i], 3) == round(control_losses[i], 3), f"Loss mismatch at step {i+1}"
        print("--> SUCCESS: HF Trainer Dual-mode resume deterministic verification passed!")

    # ----------------------------------------------------
    # 3. STRICT MODE: Interrupt at step 3, delete pickle, resume
    # ----------------------------------------------------
    model, dataset = get_reproducible_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        loss_tracker = LossTrackerCallback()
        opt_callback = OptimTensorsCallback(mode="strict")
        
        training_args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=3,
            per_device_train_batch_size=8,
            logging_steps=1,
            save_steps=3,
            seed=1337,
            data_seed=1337,
            full_determinism=True,
            report_to="none",
        )
        
        trainer = SafeTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[loss_tracker, opt_callback],
        )
        trainer.train()
        interrupted_losses = list(loss_tracker.losses)
        
        checkpoint_dir = os.path.join(tmpdir, "checkpoint-3")
        assert not os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt"))
        assert os.path.isfile(os.path.join(checkpoint_dir, "optimizer.optimtensors"))
        
        # Resume training from step 3 using SafeTrainer (loading without optimizer.pt)
        loss_tracker_resume = LossTrackerCallback()
        training_args_resume = TrainingArguments(
            output_dir=tmpdir,
            max_steps=6,
            per_device_train_batch_size=8,
            logging_steps=1,
            seed=1337,
            data_seed=1337,
            full_determinism=True,
            report_to="none",
        )
        
        trainer_resume = SafeTrainer(
            model=model,
            args=training_args_resume,
            train_dataset=dataset,
            callbacks=[loss_tracker_resume],
        )
        trainer_resume.train(resume_from_checkpoint=checkpoint_dir)
        resumed_losses = list(loss_tracker_resume.losses)
        
        # Validate strict loss curve matches control exactly to 3 decimal places
        combined_losses = interrupted_losses + resumed_losses
        for i in range(6):
            assert round(combined_losses[i], 3) == round(control_losses[i], 3), f"Strict loss mismatch at step {i+1}"
        print("--> SUCCESS: HF Trainer Strict-mode resume deterministic verification passed!")


@pytest.mark.skipif(not INTEGRATIONS_AVAILABLE, reason="Hugging Face Transformers not available")
def test_save_total_limit_rotation():
    dataset = TinyDataset(size=32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt_callback = OptimTensorsCallback(mode="dual")
        
        training_args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=4,
            per_device_train_batch_size=8,
            save_steps=1,  # Save at steps 1, 2, 3, 4
            save_total_limit=2,  # Keep only the last 2 checkpoints
            report_to="none",
        )
        
        trainer = SafeTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[opt_callback],
        )
        trainer.train()
        
        # Check directories
        dirs = sorted([d for d in os.listdir(tmpdir) if d.startswith("checkpoint-")])
        assert dirs == ["checkpoint-3", "checkpoint-4"]
        
        # Check if deleted checkpoints had their optimizer.optimtensors cleaned up
        for d in ["checkpoint-1", "checkpoint-2"]:
            assert not os.path.exists(os.path.join(tmpdir, d))
            
        # Verify current checkpoints have optimizer.optimtensors
        for d in dirs:
            assert os.path.isfile(os.path.join(tmpdir, d, "optimizer.optimtensors"))
        print("--> SUCCESS: save_total_limit rotation cleans up safe checkpoints automatically!")


@pytest.mark.skipif(not INTEGRATIONS_AVAILABLE, reason="Hugging Face Transformers not available")
def test_corrupted_file_resume_raises():
    dataset = TinyDataset(size=16)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt_callback = OptimTensorsCallback(mode="strict")
        
        training_args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=1,
            per_device_train_batch_size=8,
            save_steps=1,
            report_to="none",
        )
        
        trainer = SafeTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[opt_callback],
        )
        trainer.train()
        
        checkpoint_dir = os.path.join(tmpdir, "checkpoint-1")
        optimtensors_path = os.path.join(checkpoint_dir, "optimizer.optimtensors")
        
        # Corrupt the file: write garbage bytes
        with open(optimtensors_path, "wb") as f:
            f.write(b"GARBAGE_BYTES_THAT_FAIL_PARSING_AND_MMAP_BOUNDS")
            
        # Try to resume from checkpoint
        training_args_resume = TrainingArguments(
            output_dir=tmpdir,
            max_steps=2,
            per_device_train_batch_size=8,
            report_to="none",
        )
        
        trainer_resume = SafeTrainer(
            model=model,
            args=training_args_resume,
            train_dataset=dataset,
        )
        
        with pytest.raises(ValueError) as excinfo:
            trainer_resume.train(resume_from_checkpoint=checkpoint_dir)
            
        assert "Invalid file" in str(excinfo.value) or "Failed to parse" in str(excinfo.value) or "exceeds safety limit" in str(excinfo.value)
        print("--> SUCCESS: Corrupted safe checkpoint throws ValueError on load as promised!")
