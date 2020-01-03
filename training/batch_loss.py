def batch_loss(dataloader, model, optimizer, loss_fn, **loss_kwargs):
    print(f'Training epoch with {len(dataloader)*dataloader.batch_size} examples... ', end='')
    running_loss = 0.0
    i = 0
    for batch in dataloader:
        # forward pass
        loss = loss_fn(model, batch, **loss_kwargs)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track training loss
        running_loss += loss.data
        i+=1
    training_loss = running_loss / i
    print(f'loss is {training_loss}')
    return {'type': 'scalar', 'value': float(training_loss)}, model, optimizer
