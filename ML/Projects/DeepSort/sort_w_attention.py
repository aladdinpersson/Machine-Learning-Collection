"""
Training a Pointer Network which is a modified
Seq2Seq with attention network for the task of
sorting arrays.
"""

from torch.utils.data import (
    Dataset,
    DataLoader,
)
import random
import torch
import torch.nn as nn
import torch.optim as optim
from utils import sort_array, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard


class SortArray(Dataset):
    def __init__(self, batch_size, min_int, max_int, min_size, max_size):
        self.batch_size = batch_size
        self.min_int = min_int
        self.max_int = max_int + 1
        self.min_size = min_size
        self.max_size = max_size + 1
        self.start_tok = torch.tensor([-1]).expand(1, self.batch_size)

    def __len__(self):
        return 10000 // self.batch_size

    def __getitem__(self, index):
        size_of_array = torch.randint(
            low=self.min_size, high=self.max_size, size=(1, 1)
        )

        unsorted_arr = torch.rand(size=(size_of_array, self.batch_size)) * (
            self.max_int - self.min_int
        )
        # unsorted_arr = torch.randint(
        #    low=self.min_int, high=self.max_int, size=(size_of_array, self.batch_size)
        # )
        sorted_arr, indices = torch.sort(unsorted_arr, dim=0)

        return unsorted_arr.float(), torch.cat((self.start_tok, indices), 0)


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(1, hidden_size, num_layers)

    def forward(self, x):
        embedding = x.unsqueeze(2)
        # embedding shape: (seq_length, N, 1)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # encoder_states: (seq_length, N, hidden_size)

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, units=100):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(hidden_size + 1, hidden_size, num_layers)
        self.energy = nn.Linear(hidden_size * 2, units)
        self.fc = nn.Linear(units, 1)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        sequence_length = encoder_states.shape[0]
        batch_size = encoder_states.shape[1]

        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        energy = self.fc(energy)

        # energy: (seq_length, N, 1)
        attention = self.softmax(energy)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size), snl
        # we want context_vector: (1, N, hidden_size), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        rnn_input = torch.cat([context_vector, x.unsqueeze(0).unsqueeze(2)], dim=2)

        # rnn_input: (1, N, hidden_size)
        _, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        return attention.squeeze(2), energy.squeeze(2), hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]

        outputs = torch.zeros(target_len, batch_size, target_len - 1).to(device)
        encoder_states, hidden, cell = self.encoder(source)

        # First input will be <SOS> token
        x = target[0]
        predictions = torch.zeros(target_len, batch_size)

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            attention, energy, hidden, cell = self.decoder(
                x, encoder_states, hidden, cell
            )

            # Store prediction for current time step
            outputs[t] = energy.permute(1, 0)

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = attention.argmax(0)
            predictions[t, :] = best_guess

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs, predictions[1:, :]


### We're ready to define everything we need for training our Seq2Seq model ###
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 1000
learning_rate = 3e-5
batch_size = 32
hidden_size = 1024
num_layers = 1  # Current implementation is only for 1 layered
min_int = 1
max_int = 10
min_size = 2
max_size = 15

# Tensorboard to get nice plots etc
writer = SummaryWriter(f"runs/loss_plot2")
step = 0

encoder_net = Encoder(hidden_size, num_layers).to(device)
decoder_net = Decoder(hidden_size, num_layers).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# following is for testing the network, uncomment this if you want
# to try out a few arrays interactively
# sort_array(encoder_net, decoder_net, device)

dataset = SortArray(batch_size, min_int, max_int, min_size, max_size)
train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": step,
        }
        save_checkpoint(checkpoint)

    for batch_idx, (unsorted_arrs, sorted_arrs) in enumerate(train_loader):
        inp_data = unsorted_arrs.squeeze(0).to(device)
        target = sorted_arrs.squeeze(0).to(device)

        # Forward prop
        output, prediction = model(inp_data, target)

        # Remove output first element (because of how we did the look in Seq2Seq
        # starting at t = 1, then reshape so that we obtain (N*seq_len, seq_len)
        # and target will be (N*seq_len)
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1
