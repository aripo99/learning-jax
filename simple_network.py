import jax
import jax.numpy as jnp
from jax import random, jit, grad
from jax.experimental import optimizers

from torchvision import datasets, transforms
import numpy as np


def initialize_params(layer_sizes: list, rng: jax.random.PRNGKey) -> list:
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        rng, key = random.split(rng)
        weights = random.normal(key, (n_out, n_in)) * jnp.sqrt(2 / n_in)
        biases = jnp.zeros(n_out)
        params.append((weights, biases))
    return params


input_size = 784
hidden_size = 512
layer_sizes = [input_size, hidden_size, 1]  # for binary classification
rng = random.PRNGKey(0)
params = initialize_params(layer_sizes, rng)


def relu(x):
    return jnp.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def forward_pass(params: list, x: jnp.ndarray) -> jnp.ndarray:
    activations = x
    for w, b in params[:-1]:  # Iterate over all layers except the last
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    # For the last layer
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return sigmoid(logits)


def loss_fn(params: list, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = forward_pass(params, x)
    return -jnp.mean(y * jnp.log(preds) + (1 - y) * jnp.log(1 - preds))


step_size = 0.1
opt_init, opt_update, get_params = optimizers.sgd(step_size)


@jit
def update(i, opt_state, x, y):
    params = get_params(opt_state)
    return opt_update(i, grad(loss_fn)(params, x, y), opt_state)


def load_mnist():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    trainset = datasets.MNIST(
        "./mnist_data", download=True, train=True, transform=transform
    )
    valset = datasets.MNIST(
        "./mnist_data", download=True, train=False, transform=transform
    )

    train_images = np.array(trainset.data, dtype=np.float32)
    train_labels = np.array(trainset.targets, dtype=np.int32)
    val_images = np.array(valset.data, dtype=np.float32)
    val_labels = np.array(valset.targets, dtype=np.int32)

    train_images = train_images.reshape(train_images.shape[0], -1) / 255
    val_images = val_images.reshape(val_images.shape[0], -1) / 255

    return (train_images, train_labels), (val_images, val_labels)


def iterate_batches(data, batch_size):
    images, labels = data
    num_batches = len(images) // batch_size

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        yield images[start:end], labels[start:end]


if __name__ == "__main__":
    opt_state = opt_init(params)

    num_epochs = 10
    batch_size = 32

    (train_images, train_labels), (val_images, val_labels) = load_mnist()

    for epoch in range(num_epochs):
        for x, y in iterate_batches((train_images, train_labels), batch_size):
            x = jnp.array(x)
            y = jnp.array(y)
            opt_state = update(epoch, opt_state, x, y)
