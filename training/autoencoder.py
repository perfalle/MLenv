def vanilla(encoder, decoder, dataloader, optimizer, criterion):
    print(f'Start training with {len(dataloader)} examples')
    encoder.train()
    decoder.train()

    epochs = 1
    for epoch in range(epochs):
        running_loss = 0.0
        n = 0
        for batch in enumerate(dataloader, 0):
            print(f'Training {epoch+1} of {epochs}... ', eol='')
            i = batch[0]
            inputs = batch[1]
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            # forward pass
            code = encoder(inputs)
            outputs = decoder(code)
            loss = criterion(outputs, inputs)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track training loss
            running_loss += loss.data
            n+=1
            print(f'running loss is {running_loss / n}')

    return encoder, decoder
