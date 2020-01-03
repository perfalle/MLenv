def batch_loss(dataloader, model, loss_fn, **loss_kwargs):
    print(f'Evaluate {loss_fn.__name__} in {len(dataloader)*dataloader.batch_size} examples... ', end='')
    running_loss = 0.0
    i = 0
    for batch in dataloader:
        # forward pass
        loss = loss_fn(model, batch, **loss_kwargs)
        # track loss
        running_loss += loss.data
        i+=1
    loss = running_loss / i
    print(f'loss is {loss}')
    return float(loss)
