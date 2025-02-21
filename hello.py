import torch
import torch.nn.functional as F
import math
class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate
        return new_h, new_cell

def test_hello():
    batch_size = 3
    input_features = 4
    seq_length = 1
    input_seq = torch.randn(seq_length, batch_size, input_features)

    # Initialize hidden state and cell state
    state_size = 5
    h = torch.zeros(batch_size, state_size)
    c = torch.zeros(batch_size, state_size)

    model = LLTM(input_features, state_size)
    outputs = []
    for t in range(seq_length):
        h, c = model(input_seq[t], (h, c))
        outputs.append(h)
    outputs = torch.stack(outputs)
    print(outputs)

def test_lltm():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define dimensions
    batch_size = 8
    input_features = 32
    state_size = 64
    seq_length = 10
    
    # Create model instance
    model = LLTM(input_features, state_size)
    
    # Generate random input sequence
    input_seq = torch.randn(seq_length, batch_size, input_features)
    
    # Initialize hidden state and cell state
    h = torch.zeros(batch_size, state_size)
    c = torch.zeros(batch_size, state_size)
    
    # Process the sequence
    print("Processing sequence...")
    outputs = []
    for t in range(seq_length):
        h, c = model(input_seq[t], (h, c))
        outputs.append(h)
    
    # Stack outputs into a single tensor
    outputs = torch.stack(outputs)
    
    # Print some statistics
    print(f"\nOutput shape: {outputs.shape}")
    print(f"Output mean: {outputs.mean():.4f}")
    print(f"Output std: {outputs.std():.4f}")
    print(f"\nOutput tensor:\n{outputs}")
    
    # Test backward pass
    loss = outputs.sum()
    loss.backward()
    
    print("\nBackward pass completed successfully!")
    print(f"Weight grad shape: {model.weights.grad.shape}")
    print(f"Bias grad shape: {model.bias.grad.shape}")

if __name__ == "__main__":
    test_hello()