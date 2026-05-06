import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from torch.optim import AdamW
from tqdm import tqdm

class MedicalImageReportDataset(Dataset):
    """
    Dataset for fine-tuning CLIP on Medical Image-Report pairs.
    """
    def __init__(self, jsonl_file, tokenizer, preprocess, image_root="."):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.image_root = image_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_root, item['image'])
        image = Image.open(image_path).convert("RGB")
        
        image_tensor = self.preprocess(image)
        text_tokens = self.tokenizer([item['report']])[0]
        
        return {
            "image": image_tensor,
            "text": text_tokens
        }

class CLIPTrainer:
    """
    Orchestrates the fine-tuning of BiomedCLIP for medical domain adaptation.
    """
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", lr=1e-6):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Precision Trainer on: {self.device}")
        
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Lower LR for foundation model fine-tuning
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.05)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)

    def train(self, train_loader, val_loader=None, epochs=3, output_dir="./medical_clip_fine_tuned"):
        """Full training loop."""
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            train_loss = self.train_epoch(train_loader)
            print(f"Training Loss: {train_loss:.4f}")
            
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")
            
            self.scheduler.step()
            
        self.save_model(output_dir)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training Batch"):
            self.optimizer.zero_grad()
            
            images = batch["image"].to(self.device)
            texts = batch["text"].to(self.device)
            
            outputs = self.model(images, texts)
            image_features, text_features, logit_scale = outputs
            
            # Contrastive Loss (Symmetric)
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logits_per_image.T
            
            labels = torch.arange(len(images), device=self.device)
            loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def save_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "biomedclip_finetuned.pt"))
        print(f"Model weights saved to {output_dir}")

def generate_dummy_training_data(output_file="sample_training.jsonl"):
    """Creates a small mock dataset to test the training pipeline."""
    samples = [
        {"image": "test.png", "report": "severe spinal canal stenosis at L4-L5."},
        {"image": "test_input.png", "report": "disc protrusion with neural foraminal narrowing."},
        {"image": "test.png", "report": "normal vertebral alignment and healthy disc signal."},
        {"image": "test_input.png", "report": "facet joint hypertrophy noted at L5-S1."}
    ]
    with open(output_file, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Created dummy training file: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="sample_training.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    print("SpineAI Precision CLIP Training Pipeline")
    
    if not os.path.exists(args.train_data):
        generate_dummy_training_data(args.train_data)
        
    trainer = CLIPTrainer()
    dataset = MedicalImageReportDataset(
        args.train_data, 
        trainer.tokenizer, 
        trainer.preprocess_train
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    trainer.train(loader, epochs=args.epochs)
