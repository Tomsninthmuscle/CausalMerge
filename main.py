
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from copy import deepcopy
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import  gc
gc.collect()
torch.cuda.empty_cache()  
torch.cuda.reset_peak_memory_stats() 

def get_resnet50_cifar10(num_classes=2):
  
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

class CausalMerger:
    def __init__(self, model_old, model_new, device='cuda'):
        self.model_old = model_old.to(device)
        self.model_new = model_new.to(device)
        self.device = device
        self.causal_masks = {}        self.anchor_params = {}
    def compute_causal_masks(self, dataloader_old, dataloader_new, criterion=nn.CrossEntropyLoss(), max_steps=50):
      
        print("🔍 Phase 1: Computing gradients and generating causal masks...")
        
        m_old = deepcopy(self.model_old).train()
        m_new = deepcopy(self.model_new).train()

        grad_old_accum = {}
        grad_new_accum = {}

        for name, p in m_old.named_parameters():
            if 'weight' not in name or 'bn' in name or 'bias' in name:
                continue
            if len(p.shape) == 4:                grad_old_accum[name] = torch.zeros(p.shape[0], p.shape[1]*p.shape[2]*p.shape[3], device=self.device)
                grad_new_accum[name] = torch.zeros(p.shape[0], p.shape[1]*p.shape[2]*p.shape[3], device=self.device)
            elif len(p.shape) == 2:                grad_old_accum[name] = torch.zeros(p.shape[0], p.shape[1], device=self.device)
                grad_new_accum[name] = torch.zeros(p.shape[0], p.shape[1], device=self.device)

        iterator_old = iter(dataloader_old)
        iterator_new = iter(dataloader_new)

        for step in range(max_steps):
            try:
                x_old, y_old = next(iterator_old)
                x_new, y_new = next(iterator_new)
            except StopIteration:
                iterator_old = iter(dataloader_old)
                iterator_new = iter(dataloader_new)
                x_old, y_old = next(iterator_old)
                x_new, y_new = next(iterator_new)

            x_old, y_old = x_old.to(self.device), y_old.to(self.device)
            x_new, y_new = x_new.to(self.device), y_new.to(self.device)

            m_old.zero_grad()
            loss_old = criterion(m_old(x_old), y_old)
            loss_old.backward()
            for name, p in m_old.named_parameters():
                if name in grad_old_accum and p.grad is not None:
                    grad = p.grad.detach()
                    if len(grad.shape) == 4: grad_old_accum[name] += grad.view(grad.size(0), -1)
                    elif len(grad.shape) == 2: grad_old_accum[name] += grad

            m_new.zero_grad()
            loss_new = criterion(m_new(x_new), y_new)
            loss_new.backward()
            for name, p in m_new.named_parameters():
                if name in grad_new_accum and p.grad is not None:
                    grad = p.grad.detach()
                    if len(grad.shape) == 4: grad_new_accum[name] += grad.view(grad.size(0), -1)
                    elif len(grad.shape) == 2: grad_new_accum[name] += grad

            if (step + 1) % 10 == 0:
                print(f"   Gradient accumulation progress: {step+1}/{max_steps}")

        print("📊 Computing channel-wise cosine similarity...")
        params_ref = dict(self.model_old.named_parameters())
        
        for name in grad_old_accum:
            G_old = grad_old_accum[name]
            G_new = grad_new_accum[name]
            
            norm_old = torch.norm(G_old, p=2, dim=1)
            norm_new = torch.norm(G_new, p=2, dim=1)
            
            dot_product = torch.sum(G_old * G_new, dim=1)
            
            cos_sim = dot_product / (norm_old * norm_new + 1e-9)
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            
            mask_vals = (cos_sim + 1) / 2.0
            
            orig_shape = params_ref[name].shape
            if len(orig_shape) == 4:
                mask_tensor = mask_vals.view(-1, 1, 1, 1).expand(orig_shape)
            elif len(orig_shape) == 2:
                mask_tensor = mask_vals.view(-1, 1).expand(orig_shape)
            else:
                mask_tensor = torch.ones(orig_shape, device=self.device)
                
            self.causal_masks[name] = mask_tensor.cpu()
        print(f"✅ Causal mask computation complete! Processed {len(self.causal_masks)} layers in total.")

    def merge_models(self, base_alpha=0.5, save_path='merged_model.pth'):
     
        print("🔄 Phase 2: Executing dynamic alpha causal merging...")
        merged_model = deepcopy(self.model_old)
        merged_model.to(self.device)
        
        params_old = dict(self.model_old.named_parameters())
        params_new = dict(self.model_new.named_parameters())
        params_merged = dict(merged_model.named_parameters())
        
        with torch.no_grad():
            for name in params_old.keys():
                if name not in self.causal_masks:
                    params_merged[name].copy_(base_alpha * params_new[name] + (1 - base_alpha) * params_old[name])
                    continue
                
                w_old = params_old[name]
                w_new = params_new[name]
                mask = self.causal_masks[name].to(self.device)
                
                confidence = mask.mean().item()
                
                if confidence > 0.8:
                    current_alpha = 0.9
                elif confidence > 0.5:
                    current_alpha = 0.5
                else:
                    current_alpha = 0.1
                
                delta = w_new - w_old
                update = delta * mask * current_alpha
                params_merged[name].copy_(w_old + update)

        torch.save(merged_model.state_dict(), save_path)
        print(f"💾 Merged model saved to: {save_path}")
        return merged_model

    def fine_tune(self, model, train_loader, val_loader=None, epochs=10, lr=1e-4, save_path='final_model.pth'):
      
        print("🔥 Phase 3: Starting causal mask regularization fine-tuning...")
        device = next(model.parameters()).device
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        criterion_task = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        anchor_params = {k: v.clone().detach() for k, v in model.named_parameters()}
        
        penalty_coeffs = {}
        for name, mask in self.causal_masks.items():
            penalty_coeffs[name] = (1.0 - mask.to(device))

        lambda_cmr = 0.5
        best_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss_task = criterion_task(outputs, targets)
                
                loss_cmr = 0.0
                count = 0
                for name, param in model.named_parameters():
                    if name in penalty_coeffs:
                        diff = param - anchor_params[name]
                        loss_cmr += torch.sum(penalty_coeffs[name] * (diff ** 2))
                        count += 1
                
                if count > 0:
                    loss_cmr = loss_cmr / count                
                loss_total = loss_task + lambda_cmr * loss_cmr
                loss_total.backward()
                optimizer.step()
                
                running_loss += loss_total.item()
            
            if val_loader is not None:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                acc = 100. * correct / total
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), save_path)
                
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")
            else:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")
            
            scheduler.step()
            
        print(f"✅ Fine-tuning complete! Best model saved.")
        return model

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device: {device}")

    print("📂 Loading local pre-trained model...")
    model_old = get_resnet50_cifar10(num_classes=2)
    model_new = get_resnet50_cifar10(num_classes=2)
    
    old_model_path = 'resnet50_car_old.pth'
    new_model_path = 'resnet50_car_new.pth'
    
    model_old.load_state_dict(torch.load(old_model_path, map_location=device))
    model_new.load_state_dict(torch.load(new_model_path, map_location=device))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_root_old = r'C:\Users\admin\Desktop\houxuexi\data\car_old'
    data_root_new = r'C:\Users\admin\Desktop\houxuexi\data\car_new'

    dataset_old_train = datasets.ImageFolder(os.path.join(data_root_old, 'train'), transform_train)
    dataset_old_test = datasets.ImageFolder(os.path.join(data_root_old, 'test'), transform_test)
    dataset_new_train = datasets.ImageFolder(os.path.join(data_root_new, 'train'), transform_train)
    dataset_new_test = datasets.ImageFolder(os.path.join(data_root_new, 'test'), transform_test)

    loader_old = DataLoader(dataset_old_train, batch_size=32, shuffle=True, num_workers=0)
    loader_new = DataLoader(dataset_new_train, batch_size=32, shuffle=True, num_workers=0)
    
    full_train = ConcatDataset([dataset_old_train, dataset_new_train])
    loader_finetune = DataLoader(full_train, batch_size=32, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset_new_test, batch_size=32, shuffle=False, num_workers=0)

    print(f"✅ Data loading complete. Old training set: {len(dataset_old_train)}, New training set: {len(dataset_new_train)}")

    merger = CausalMerger(model_old, model_new, device)
    
    merger.compute_causal_masks(loader_old, loader_new, max_steps=50)
    
    merged_model = merger.merge_models(base_alpha=0.5, save_path='merged_causal.pth')
    
    final_model = merger.fine_tune(
        merged_model, 
        train_loader=loader_finetune, 
        val_loader=loader_test, 
        epochs=20, 
        lr=1e-4, 
        save_path='best_causal_model.pth'
    )

if __name__ == "__main__":
    main()
