import torch


def ask_user():
    print("Write your array as a list [i,j,k..] with arbitrary positive numbers")
    array = input("Input q if you want to quit \n")
    return array


def sort_array(encoder, decoder, device, arr=None):
    """
    A very simple example of use of the model
    Input: encoder nn.Module
           decoder nn.Module
           device
           array to sort (optional)
    """

    if arr is None:
        arr = ask_user()

    with torch.no_grad():
        while arr != "q":
            # Avoid numerical errors by rounding to max_len
            arr = eval(arr)
            lengths = [
                len(str(elem).split(".")[1]) if len(str(elem).split(".")) > 1 else 0
                for elem in arr
            ]
            max_len = max(lengths)
            source = torch.tensor(arr, dtype=torch.float).to(device).unsqueeze(1)
            batch_size = source.shape[1]
            target_len = source.shape[0] + 1

            outputs = torch.zeros(target_len, batch_size, target_len - 1).to(device)
            encoder_states, hidden, cell = encoder(source)

            # First input will be <SOS> token
            x = torch.tensor([-1], dtype=torch.float).to(device)
            predictions = torch.zeros((target_len)).to(device)

            for t in range(1, target_len):
                # At every time step use encoder_states and update hidden, cell
                attention, energy, hidden, cell = decoder(
                    x, encoder_states, hidden, cell
                )

                # Store prediction for current time step
                outputs[t] = energy.permute(1, 0)

                # Get the best word the Decoder predicted (index in the vocabulary)
                best_guess = attention.argmax(0)
                predictions[t] = best_guess.item()
                x = torch.tensor([best_guess.item()], dtype=torch.float).to(device)

            output = [
                round(source[predictions[1:].long()][i, :].item(), max_len)
                for i in range(source.shape[0])
            ]

            print(f"Here's the result: {output}")
            arr = ask_user()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):  # , steps):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # steps = checkpoint['steps']
    # return steps
