import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# Create a simple toy dataset example, normally this
# would be doing custom class with __getitem__ etc,
# which we have done in custom dataset tutorials
x = torch.randn((1000, 3, 224, 224))
y = torch.randint(low=0, high=10, size=(1000, 1))
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=8)


model = nn.Sequential(
    nn.Conv2d(3, 10, kernel_size=3, padding=1, stride=1),
    nn.Flatten(),
    nn.Linear(10*224*224, 10),
)

NUM_EPOCHS = 100
for epoch in range(NUM_EPOCHS):
    loop = tqdm(loader)
    for idx, (x, y) in enumerate(loop):
        scores = model(x)

        # here we would compute loss, backward, optimizer step etc.
        # you know how it goes, but now you have a nice progress bar
        # with tqdm

        # then at the bottom if you want additional info shown, you can
        # add it here, for loss and accuracy you would obviously compute
        # but now we just set them to random values
        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=torch.rand(1).item(), acc=torch.rand(1).item())

# There you go. Hope it was useful :)




