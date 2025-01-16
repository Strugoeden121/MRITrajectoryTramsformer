import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import BasicTransformerShot, BasicTransformerTraj, BasicTransformerShotAllFrames
from utils.constants import INPUT_NUM_FRAMES, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, IN_CHANNELS, COMPLEX_DIM


# Dummy dataset generation (replace with real data)
def generate_dummy_data(num_samples):
    inputs = torch.rand(num_samples, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, COMPLEX_DIM)  # Example shape
    targets = torch.rand(num_samples, INPUT_NUM_SHOTS, INPUT_NUM_SAMPLES, COMPLEX_DIM)  # Example shape
    return inputs, targets

train_inputs, train_targets = generate_dummy_data(10)
train_dataset = TensorDataset(train_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=INPUT_NUM_FRAMES, shuffle=True)

def train(model, dataloader, criterion, optimizer, epochs=10, device="cpu"):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device if using GPU
            inputs, targets = inputs, targets

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}")

def train_autoregressive_trajectory(model, dataloader, criterion, optimizer, epochs=10, device="cpu"):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device if using GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize variables
            batch_size = inputs.size(0)
            predictions = torch.zeros_like(targets)  # To store predictions for the entire sequence

            # Initialize the autoregressive process with the first frame of the input
            optimizer.zero_grad()  # Clear gradients

            output = model(inputs, batch_size)

            # Compute loss
            loss = criterion(output, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}")

def train_autoregressive_shot(model, dataloader, criterion, optimizer, epochs=10, device="cpu", only_last_frame=False):
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device if using GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize variables
            batch_size = inputs.size(0)
            predictions = torch.zeros((8,16,513,2))  # To store predictions for the entire sequence

            # Initialize the autoregressive process with the first frame of the input
            current_input = inputs.clone()  # Start with the original input

            optimizer.zero_grad()  # Clear gradients
            current_input = current_input.permute(0, 2, 3, 1)

            # Autoregressive loop over frames
            for t in range(INPUT_NUM_FRAMES):
                # Forward pass
                if only_last_frame:
                    output = model(current_input)
                else:
                    output = model(current_input, t+1)

                output = output.permute(1, 2, 3, 0)

                # Store prediction
                predictions[t, :, :, :] = output[0, :, :, :]

                # Update current_input for next time step
                if only_last_frame or t == 0:
                    current_input = output
                else:
                    current_input = torch.cat((current_input, output), dim=0)

            # Compute loss
            loss = criterion(predictions, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {epoch_loss/len(dataloader):.4f}")

# Model, loss, and optimizer
if __name__ == '__main__':
    model = BasicTransformerShotAllFrames()
    # model = BasicTransformerShot()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    # train_autoregressive_shot(model, train_loader, criterion, optimizer, epochs=10)
    train_autoregressive_shot(
        model, train_loader, criterion, optimizer, epochs=10)

    # Save the model
    torch.save(model.state_dict(), "basic_transformer_model.pth")
    print("Training complete. Model saved.")
