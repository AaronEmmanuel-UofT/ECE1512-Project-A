"""Training and evaluation functions shared with the completed notebook."""
import torch
import torch.nn.functional as F
from utils import classification_metrics

def train_one_epoch(model, criterion, data_loader, optimizer, device):
    model.train()
    total_loss = total_correct = total_examples = 0
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device, dtype=torch.long)
        mask = data['mask'].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(bag, mask=mask)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        count = labels.shape[0]
        total_loss += loss.item() * count
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += count
    if not total_examples:
        raise ValueError('Empty training set')
    return {'loss': total_loss / total_examples,
            'acc': 100.0 * total_correct / total_examples}

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf):
    model.eval()
    total_loss = total_examples = 0
    probabilities_all, labels_all = [], []
    for data in data_loader:
        bag = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device, dtype=torch.long)
        logits = model(bag, mask=data['mask'].to(device))
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.shape[0]
        total_examples += labels.shape[0]
        probabilities_all.append(F.softmax(logits, dim=1).cpu())
        labels_all.append(labels.cpu())
    if not total_examples:
        raise ValueError('Empty evaluation set')
    probabilities = torch.cat(probabilities_all)
    labels = torch.cat(labels_all)
    # Undefined AUROC must not silently select an invalid checkpoint.
    if labels.unique().numel() != int(conf.n_class):
        raise ValueError('Evaluation split lacks a class; macro AUROC is undefined')
    metrics = classification_metrics(probabilities, labels, conf.n_class)
    metrics['loss'] = total_loss / total_examples
    return metrics
