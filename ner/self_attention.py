import torch
from torch.nn.functional import softmax


def tensor_and_print(input):
    if isinstance(input, set):
        new_list = []
        for x in input:
            x = torch.tensor(x, dtype=torch.float32)
            print("{}".format(x))
            new_list.append(x)
        out = set(new_list)
    else:
        out = torch.tensor(input, dtype=torch.float32)
        print("{}".format(out))
    return out


# Step 1: Prepare inputs
x = [
    [1, 0, 1, 0],  # Input 1
    [0, 2, 0, 2],  # Input 2
    [1, 1, 1, 1]   # Input 3
]
x = tensor_and_print(x)
print("Input tensor size: {}".format(x.size()))

# Step 2: Initialise weights
w_key = [
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 0]
]
w_query = [
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1]
]
w_value = [
    [0, 2, 0],
    [0, 3, 0],
    [1, 0, 3],
    [1, 1, 0]
]
w_key, w_query, w_value = tensor_and_print((w_key, w_query, w_value))
print("w_key size: {}".format(w_key.size()))


# Step 3: Derive key, query and value
keys = x @ w_key
queries = x @ w_query
values = x @ w_value
# torch.mm: https://pytorch.org/docs/stable/torch.html#torch.mm
# torch.matmul: https://pytorch.org/docs/stable/torch.html#torch.matmul
# @ == matmul? for example, print(keys@torch.tensor([1.0, 1.0, 1.0]))

print(keys, keys.size())
# tensor([[0., 1., 1.],  # input1
#         [4., 4., 0.],  # input2
#         [2., 3., 1.]]) # input3

print(queries)
# tensor([[1., 0., 2.],
#         [2., 2., 2.],
#         [2., 1., 3.]])

print(values)
# tensor([[1., 2., 3.],
#         [2., 8., 0.],
#         [2., 6., 3.]])


# Step 4: Calculate attention scores
attn_scores = queries @ keys.T
# here key.T let the first input of keys is multiplicated! so the score = query_input1*key_input1

# tensor([[ 2.,  4.,  4.],  # attention scores from Query 1
#         [ 4., 16., 12.],  # attention scores from Query 2
#         [ 4., 12., 10.]]) # attention scores from Query 3


# Step 5: Calculate softmax
attn_scores_softmax = softmax(attn_scores, dim=-1)
# tensor([[6.3379e-02, 4.6831e-01, 4.6831e-01], # first query * all keys of inputs
#         [6.0337e-06, 9.8201e-01, 1.7986e-02],
#         [2.9539e-04, 8.8054e-01, 1.1917e-01]])

### how to calculate the weighted value?

# motivation: [[[x, x, x], # 2.0 * values[1]
# ],
# ]
weighted_value = attn_scores_softmax @ values.T


# For readability, approximate the above as follows
attn_scores_softmax = [
    [0.0, 0.5, 0.5],
    [0.0, 1.0, 0.0],
    [0.0, 0.9, 0.1]
]
attn_scores_softmax = torch.tensor(attn_scores_softmax)

# Step 6: Multiply scores with values
weighted_values = values[:, None] * attn_scores_softmax.T[:, :, None]

# tensor([[[0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000],
#          [0.0000, 0.0000, 0.0000]],
#
#         [[1.0000, 4.0000, 0.0000],
#          [2.0000, 8.0000, 0.0000],
#          [1.8000, 7.2000, 0.0000]],
#
#         [[1.0000, 3.0000, 1.5000],
#          [0.0000, 0.0000, 0.0000],
#          [0.2000, 0.6000, 0.3000]]])

# Step 7: Sum weighted values
outputs = weighted_values.sum(dim=0)

# tensor([[2.0000, 7.0000, 1.5000],  # Output 1
#         [2.0000, 8.0000, 0.0000],  # Output 2
#         [2.0000, 7.8000, 0.3000]]) # Output 3

"""
Note:
PyTorch has provided an API for this called nn.MultiheadAttention. However, this API requires that you feed in key,
query and value PyTorch tensors. Moreover, the outputs of this module undergo a linear transformation.
"""