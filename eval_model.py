def model_loss(model, data, criterion, device):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    preds = model(inputs.float())
    loss = criterion(preds, labels).item()
    return loss


def baseline_model_loss(data, criterion, lag=18):
    pred = data[1][:-lag]
    actual = data[1][lag:]
    loss = criterion(pred, actual).item()
    return loss
