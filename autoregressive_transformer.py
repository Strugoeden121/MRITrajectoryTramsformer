import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


# Define the Autoregressive Transformer Model without Encoder
import torch
import torch.nn as nn


class AutoregressiveTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads, num_layers, device):
        super(AutoregressiveTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.device = device

        # Transformer model configuration
        self.transformer = nn.Transformer(
            batch_first=True,
            d_model=feature_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=0.1
        )

        # Embedding layer for input sequences
        self.embedding = nn.Linear(feature_dim, feature_dim)

        # Positional encoding (can be replaced with other types of encodings if needed)
        self.positional_encoding = nn.Embedding(100, feature_dim)  # Max seq length for positional encoding

        # Final output layer
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def forward(self, src, tgt):
        # Embed and add positional encoding
        src_emb = self.embedding(src) + self.positional_encoding(torch.arange(src.size(1), device=self.device))
        tgt_emb = self.embedding(tgt) + self.positional_encoding(torch.arange(tgt.size(1), device=self.device))

        # Pass through transformer (this applies the same transformer to both src and tgt)
        output = self.transformer(src_emb, tgt_emb)  # No mask applied here

        # Output layer
        output = self.fc_out(output)
        return output

    def autoregressive_inference(self, initial_input, max_steps):
        generated = [initial_input]  # List to store all inputs and predictions
        for step in range(max_steps-1):
            print(f"Step {step + 1}")
            current_input = torch.cat(generated, dim=1)  # Concatenate along seq_len
            print(f"Current input shape: {current_input.shape}")
            output = self(current_input, torch.zeros(8,1,64))  # Same input for both src and tgt in inference
            print(f"Output shape: {output.shape}")
            next_prediction = output[:, -1:, :]  # Take the last output
            generated.append(next_prediction)  # Append prediction to the sequence

        return torch.cat(generated, dim=1)  # Final concatenated output


# Training function outside the model
def train(model, train_loader, criterion, optimizer, device, num_epochs=10, max_steps=8):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            print(f"Batch {batch_idx}, Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")

            # Forward pass
            optimizer.zero_grad()
            output = model.autoregressive_inference(inputs, max_steps)

            # Compute loss (mean squared error or any other criterion)
            loss = criterion(output, targets)
            loss.backward()

            # Update model parameters
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss}")


# Main function to run everything
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    feature_dim = 64  # Reduced number of features
    batch_size = 8
    num_heads = 8
    num_layers = 4
    num_epochs = 10
    max_steps = 8

    # Initialize the model and optimizer
    model = AutoregressiveTransformer(feature_dim, num_heads, num_layers, device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy dataset (replace with real data for actual training)
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size, feature_dim):
            self.data = torch.randn(size, 1, feature_dim)
            self.targets = torch.randn(size, 8, feature_dim)  # Target sequence length 8

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    train_loader = torch.utils.data.DataLoader(DummyDataset(100, feature_dim), batch_size=batch_size, shuffle=True)

    # Loss function (Mean Squared Error loss)
    criterion = nn.MSELoss()

    # Train the model
    print("Training started...")
    train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs, max_steps=max_steps)


if __name__ == "__main__":
    main()
