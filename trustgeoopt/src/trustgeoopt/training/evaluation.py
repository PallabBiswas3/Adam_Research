import torch


@torch.no_grad()
def evaluate(model, loader, criterion, device, task="classification"):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prediction = model(x)
        total_loss += float(criterion(prediction, y))*x.shape[0]
        if task == "classification":
            correct += int((prediction.argmax(1) == y).sum())
        count += x.shape[0]
    if count == 0:
        raise ValueError("empty evaluation loader")
    return dict(loss=total_loss/count, accuracy=correct/count if task == "classification" else None)
