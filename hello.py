import torch
import torch.nn.functional as F
import math
import lltm_cpp

class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell

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
        return LLTMFunction.apply(input, self.weights, self.bias, *state)

    # def forward(self, input, state):
    #     old_h, old_cell = state
    #     X = torch.cat([old_h, input], dim=1)

    #     # Compute the input, output and candidate cell gates with one MM.
    #     gate_weights = F.linear(X, self.weights, self.bias)
    #     # Split the combined gate weight matrix into its components.
    #     gates = gate_weights.chunk(3, dim=1)

    #     input_gate = torch.sigmoid(gates[0])
    #     output_gate = torch.sigmoid(gates[1])
    #     # # Here we use an ELU instead of the usual tanh.
    #     candidate_cell = F.elu(gates[2])

    #     # Compute the new cell state.
    #     new_cell = old_cell + candidate_cell * input_gate
    #     # Compute the new hidden state and output.
    #     new_h = torch.tanh(new_cell) * output_gate
    #     return new_h, new_cell

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

    import time

    import torch

    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    batch_size = 16
    input_features = 32
    state_size = 128

    X = torch.randn(batch_size, input_features)
    h = torch.randn(batch_size, state_size)
    C = torch.randn(batch_size, state_size)

    rnn = LLTM(input_features, state_size)

    forward = 0
    backward = 0
    for _ in range(100000):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start

    print('Forward: {:.3f} s | Backward {:.3f} s'.format(forward, backward))