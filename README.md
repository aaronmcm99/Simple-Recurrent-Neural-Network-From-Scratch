# Simple-Recurrent-Neural-Network-From-Scratch

This implementation describes a neural network used for predicting time-series data. Specifically, the model predicts the temperature for the next day based on features such as the current day's maximum temperature, minimum temperature, and rainfall. The core of this model is built using Recurrent Neural Networks (RNNs) with forward and backward passes.

## Data Preprocessing
First, the data is loaded and preprocessed to scale the predictors (features) to have zero mean using StandardScaler. The dataset is then split into training, validation, and test sets.
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define predictors and target
PREDICTORS = ["tmax", "tmin", "rain"]
TARGET = "tmax_tomorrow"

# Scale the data to have mean 0
scaler = StandardScaler()
data[PREDICTORS] = scaler.fit_transform(data[PREDICTORS])

# Split the dataset into training, validation, and test sets
np.random.seed(0)
split_data = np.split(data, [int(0.7 * len(data)), int(0.85 * len(data))])
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [
    [d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in split_data
]
```

## Initializing Parameters
We initialize the weights and biases for the neural network. The initialization is done with small random values using the standard method of scaling by the inverse square root of the number of units in each layer.
```python
import math

def init_params(layer_conf):
    layers = []
    for i in range(1, len(layer_conf)):
        np.random.seed(0)
        k = 1 / math.sqrt(layer_conf[i]["hidden"])
        i_weight = np.random.rand(layer_conf[i-1]["units"], layer_conf[i]["hidden"]) * 2 * k - k
        h_weight = np.random.rand(layer_conf[i]["hidden"], layer_conf[i]["hidden"]) * 2 * k - k
        h_bias = np.random.rand(1, layer_conf[i]["hidden"]) * 2 * k - k
        o_weight = np.random.rand(layer_conf[i]["hidden"], layer_conf[i]["output"]) * 2 * k - k
        o_bias = np.random.rand(1, layer_conf[i]["output"]) * 2 * k - k
        layers.append([i_weight, h_weight, h_bias, o_weight, o_bias])
    return layers

```

## Forward Pass
The forward pass computes the hidden states and outputs of the network. For each time step, the input is passed through the network's layers, and the output is generated. The hidden states are stored for use in the backpropagation step.
```python
def forward(x, layers):
    hiddens = []
    outputs = []
    for i in range(len(layers)):
        i_weight, h_weight, h_bias, o_weight, o_bias = layers[i]
        hidden = np.zeros((x.shape[0], i_weight.shape[1]))
        output = np.zeros((x.shape[0], o_weight.shape[1]))
        for j in range(x.shape[0]):
            input_x = x[j,:][np.newaxis,:] @ i_weight
            hidden_x = input_x + hidden[max(j-1,0),:][np.newaxis,:] @ h_weight + h_bias
            hidden_x = np.tanh(hidden_x)  # Apply tanh activation
            hidden[j,:] = hidden_x
            output_x = hidden_x @ o_weight + o_bias
            output[j,:] = output_x
        hiddens.append(hidden)
        outputs.append(output)
    return hiddens, outputs[-1]
```

## Backward Pass
The backward pass updates the weights and biases by calculating the gradients using backpropagation through time (BPTT). The gradients are computed at each time step, and the weights are updated using the learning rate.
```python
def backward(layers, x, lr, grad, hiddens):
    for i in range(len(layers)):
        i_weight, h_weight, h_bias, o_weight, o_bias = layers[i]
        hidden = hiddens[i]
        next_h_grad = None
        i_weight_grad, h_weight_grad, h_bias_grad, o_weight_grad, o_bias_grad = [0] * 5
        for j in range(x.shape[0] - 1, -1, -1):
            out_grad = grad[j,:][np.newaxis, :]
            o_weight_grad += hidden[j,:][:, np.newaxis] @ out_grad
            o_bias_grad += out_grad
            h_grad = out_grad @ o_weight.T
            if j < x.shape[0] - 1:
                hh_grad = next_h_grad @ h_weight.T
                h_grad += hh_grad
            tanh_deriv = 1 - hidden[j][np.newaxis,:] ** 2
            h_grad = np.multiply(h_grad, tanh_deriv)
            next_h_grad = h_grad.copy()
            if j > 0:
                h_weight_grad += hidden[j-1][:, np.newaxis] @ h_grad
                h_bias_grad += h_grad
            i_weight_grad += x[j,:][:,np.newaxis] @ h_grad
        lr = lr / x.shape[0]  # Normalize by sequence length
        i_weight -= i_weight_grad * lr
        h_weight -= h_weight_grad * lr
        h_bias -= h_bias_grad * lr
        o_weight -= o_weight_grad * lr
        o_bias -= o_bias_grad * lr
        layers[i] = [i_weight, h_weight, h_bias, o_weight, o_bias]
    return layers
```

## Training Loop
The training loop runs for a predefined number of epochs. In each epoch, the model is trained on sequences of the input data, and the loss is calculated using the mean squared error (MSE) function. The model is updated using the gradients calculated during the backward pass.
```python
epochs = 250
lr = 1e-5

layer_conf = [
    {"type": "input", "units": 3},
    {"type": "rnn", "hidden": 4, "output": 1}
]
layers = init_params(layer_conf)

for epoch in range(epochs):
    sequence_len = 7
    epoch_loss = 0
    for j in range(train_x.shape[0] - sequence_len):
        seq_x = train_x[j:(j+sequence_len),]
        seq_y = train_y[j:(j+sequence_len),]
        hiddens, outputs = forward(seq_x, layers)
        grad = mse_grad(seq_y, outputs)  # Compute the gradient of the MSE loss
        params = backward(layers, seq_x, lr, grad, hiddens)  # Update weights
        epoch_loss += mse(seq_y, outputs)  # Add loss to epoch loss

    if epoch % 50 == 0:
        sequence_len = 7
        valid_loss = 0
        for j in range(valid_x.shape[0] - sequence_len):
            seq_x = valid_x[j:(j+sequence_len),]
            seq_y = valid_y[j:(j+sequence_len),]
            _, outputs = forward(seq_x, layers)
            valid_loss += mse(seq_y, outputs)
        
        print(f"Epoch: {epoch} train loss {epoch_loss / len(train_x)} valid loss {valid_loss / len(valid_x)}")
```

## Loss Function
The mean squared error (MSE) loss function and its gradient are used to evaluate and update the model's parameters.
```python
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    return -2 * (y_true - y_pred) / y_true.shape[0]
```
