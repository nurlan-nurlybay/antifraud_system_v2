# Parameter Tuning Algorithms

This note explains two related but different ideas:

- **Gradient descent** is how we tune model weights.
- **Optuna / TPE** is how we search for good hyperparameters.

---

## Gradient Descent: moving downhill

Gradient descent updates weights by moving in the opposite direction of the loss gradient.

When training with mini-batches, a typical loop is:

1. Split the data into a small batch, e.g. 1024 samples.
2. Compute the gradient of the loss for each weight over that batch.
3. Average those gradients to get a stable direction.
4. Move each weight opposite to the gradient, scaled by the learning rate.

This is a compromise between pure SGD (one example at a time) and full batch gradient descent (all examples at once).

The weight update formula is:

    w_{t+1} = w_t - η * avg_gradient

Where:

- `w_t` is the current weight.
- `η` is the learning rate.
- `avg_gradient` is the average gradient over the batch.

---

## Optuna, Bayesian Optimizer, TPE: tuning hyperparameters

Optuna is a hyperparameter optimizer. The TPE algorithm inside Optuna works by modeling two distributions:

- a **good** distribution of values from the best trials
- a **bad** distribution of values from the worst trials

Then it chooses new hyperparameter values that are much more likely under the good distribution than under the bad one.

That is, TPE looks for values that appear often in successful trials and rarely in unsuccessful ones.

---

## Adam optimizer: SGD + momentum + RMSProp

Adam combines the stability of momentum with per-parameter scaling from RMSProp.

### RMSProp intuition

SGD uses the same learning rate `η` for every weight. RMSProp changes that by tracking how large each weight's gradients are:

- If a parameter has huge, noisy gradients, RMSProp divides the step by a large number, slowing the update.
- If a parameter has tiny gradients, it divides by a small number, making the step larger.

This means each parameter effectively gets its own adaptive step size.

### Adam formulas

Adam keeps two moving averages for each parameter:

    m = 0.9 * m + 0.1 * avg_gradient
    v = 0.999 * v + 0.001 * (avg_gradient ** 2)

Then it updates weights as:

    w = w - η * (m / sqrt(v))

Here:

- `m` is the momentum-like average of gradients.
- `v` is the running average of squared gradients.
- `sqrt(v)` scales the step by gradient magnitude.

### Why this helps

- For very small gradients, `sqrt(v)` is small, so the effective step size becomes larger. This helps tune flat directions instead of making tiny, useless updates.
- For medium gradients, the algorithm behaves more like normal SGD, letting the slope do the work.
- For extremely large gradients, the denominator grows, damping the update so the parameter does not bounce wildly.

In other words, Adam adjusts its effective learning rate per parameter: larger when gradients are tiny, smaller when gradients are huge.

---

### P.S. Some people do this

Some people use SGD + RMSprop, which is Adam-momentum.

    v = 0.999 * v + 0.001 * (avg_gradient ** 2)

Then it updates weights as:

    w = w - η * (avg_gradient / sqrt(v))

so in SGD + RMSprop I just replace "m" (moving / smoothed average gradient) with average gradient itself