def model_loss(model, data, criterion):
    inputs, labels = data
    preds = model(inputs.float())
    loss = criterion(preds.double(), labels.double()).item()
    return loss


def baseline_model_loss(data, criterion, lag=18):
    pred = data[1][:-lag]
    actual = data[1][lag:]
    loss = criterion(pred.double(), actual.double()).item()
    return loss
