import torch
import torch.nn.functional as F

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
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell


# Test 1: Basic functionality
def test_simple():
    print("Test 1: Basic functionality")
    X = torch.tensor([[1., 2.]], dtype=torch.float32)
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    bias = torch.tensor([0.1, 0.2], dtype=torch.float32)

    gate_weights = F.linear(X, weights, bias)
    print(f"Input X: {X}")
    print(f"Weights: {weights}")
    print(f"Bias: {bias}")
    print(f"Output: {gate_weights}")
    print(f"Output shape: {gate_weights.shape}\n")

# Test 2: Batch processing
def test_batch():
    print("Test 2: Batch processing")
    X = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    bias = torch.tensor([0.1, 0.2], dtype=torch.float32)

    gate_weights = F.linear(X, weights, bias)
    print(f"Batch Input X: {X}")
    print(f"Output: {gate_weights}")
    print(f"Output shape: {gate_weights.shape}\n")

# Test 3: With random values
def test_random():
    print("Test 3: Random values")
    batch_size = 2
    input_features = 4
    output_features = 3
    
    X = torch.randn(batch_size, input_features)
    weights = torch.randn(output_features, input_features)
    bias = torch.randn(output_features)

    gate_weights = F.linear(X, weights, bias)
    print(f"Input shape: {X.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Output shape: {gate_weights.shape}")
    print(f"Output: {gate_weights}\n")

if __name__ == "__main__":
    test_simple()
    # test_batch()
    # test_random()