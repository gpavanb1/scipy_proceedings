---
title: "NODEFit: Fit time-series data with a Neural Differential Equation"
abstract: |
  Time-series data often arise from underlying physical, biological, and engineering systems. Traditional regression models frequently ignore the continuous dynamics governing these processes. Neural Ordinary Differential Equations (Neural ODEs) and Neural Stochastic Differential Equations (Neural SDEs) provide a powerful framework for learning these evolution laws directly from observations. This paper introduces NODEFit, an open-source Python package that provides a simple and efficient interface for fitting ODEs and SDEs to measured data. We demonstrate the capabilities of NODEFit in capturing complex dynamics and providing robust extrapolations.
---

## Introduction

Physical phenomena are commonly governed by differential equations. Traditional methods for time-series analysis often rely on discrete-time models, which may fail to capture the underlying continuous-time dynamics. Neural Ordinary Differential Equations (Neural ODEs) [@chen2018torchdiffeq] represent a paradigm shift by modeling the latent state evolution as a continuous process.

NODEFit provides a simple Python interface for fitting ODEs and SDEs to measured data, allowing researchers to fit their data directly to the governing physical laws.

## Methods

### Continuous-time model

We model the latent state evolution as:

```{math}
:label: eq:node
\frac{dy(t)}{dt} = f_\theta(y(t), t)
```

where $f_\theta$ is a neural network representing the unknown dynamics.

### Stochastic dynamics

For noisy systems, NODEFit supports Neural SDEs [@li2020scalable; @kidger2021neural]:

```{math}
:label: eq:nsde
dy = f_\theta(y, t) dt + g_\theta(y, t) dW_t
```

where $W_t$ is a Wiener process and $g_\theta$ is a neural network representing the diffusion term.

### Training Objective

Given observations $(t_i, y_i)$, NODEFit minimizes the mean squared error:

```{math}
:label: eq:loss
\mathcal{L} = \sum_i ||y_i - \hat{y}(t_i)||^2
```

where $\hat{y}(t)$ is produced by the ODE/SDE solver.

### Backpropagation

A differentiable ODE solver computes the forward trajectory, and gradients are propagated through the integration process. To derive the adjoint equation from first principles, consider a small time step $\epsilon$. Using a first-order Taylor expansion:

```{math}
:label: eq:taylor
y(t + \epsilon) = y(t) + \epsilon f(y(t), t, \theta) + O(\epsilon^2)
```

By the chain rule, the sensitivity of the loss $\mathcal{L}$ with respect to the state at time $t$, defined as the adjoint state $a(t) = \partial \mathcal{L} / \partial y(t)$, is:

```{math}
:label: eq:chain_rule
a(t) = \frac{\partial \mathcal{L}}{\partial y(t)} = \frac{\partial \mathcal{L}}{\partial y(t+\epsilon)} \frac{\partial y(t+\epsilon)}{\partial y(t)}
```

Substituting the expansion from @eq:taylor into @eq:chain_rule:

```{math}
:label: eq:subst
a(t) = a(t+\epsilon) \left( I + \epsilon \frac{\partial f(y(t), t, \theta)}{\partial y(t)} \right)
```

Rearranging and taking the limit $\epsilon \to 0$ yields the adjoint ODE:

```{math}
:label: eq:adjoint_deriv
\frac{da(t)}{dt} = -a(t)^T \frac{\partial f(y(t), t, \theta)}{\partial y}
```

This enables gradient computation with constant memory cost by solving this equation backwards in time.

## Implementation

NODEFit is implemented as an open-source Python package built on top of the PyTorch ecosystem. It leverages specialized libraries to handle the numerical integration and gradient computation required for training Neural ODEs and SDEs.

### Core Dependencies

The efficiency and scalability of NODEFit rely on two primary libraries: `torchdiffeq` and `torchsde`.

#### torchdiffeq
For Ordinary Differential Equations, NODEFit utilizes `torchdiffeq` [@chen2018torchdiffeq]. This library provides a suite of differentiable ODE solvers. A critical feature of `torchdiffeq` is its support for the **adjoint sensitivity method**. Unlike standard backpropagation through the solver's internal operations (which has a memory cost that scales with the number of solver steps), the adjoint method allows for gradient computation with constant memory cost by solving an augmented ODE backwards in time. The adjoint state $a(t) = \partial \mathcal{L} / \partial y(t)$ follows the dynamics:

```{math}
:label: eq:adjoint
\frac{da(t)}{dt} = -a(t)^T \frac{\partial f(y(t), t, \theta)}{\partial y}
```

The gradient with respect to the model parameters $\theta$ is then computed as:

```{math}
:label: eq:grad_theta
\frac{d\mathcal{L}}{d\theta} = -\int_{t_1}^{t_0} a(t)^T \frac{\partial f(y(t), t, \theta)}{\partial \theta} dt
```

This enables the training of complex models on large time-series datasets that would otherwise be computationally prohibitive.

#### torchsde
For stochastic systems, NODEFit integrates `torchsde` [@li2020scalable]. Stochastic Differential Equations present unique challenges, particularly in ensuring consistent Brownian motion across multiple steps and handling the nuances of stochastic calculus. 

To derive the stochastic adjoint from first principles, we consider the SDE in Stratonovich form (denoted by the $\circ$ operator). The Stratonovich integral evaluates the integrand at the midpoint of the interval, $g(y, t) \circ dW_t \approx g(y_{t+\Delta t/2}, t+\Delta t/2) \Delta W_t$. This choice ensures that the SDE obeys the standard rules of calculus:

```{math}
:label: eq:strat_sde
dy = f(y, t, \theta) dt + g(y, t, \theta) \circ dW_t
```

Consider a small time step $\Delta t$. The state update is approximately:

```{math}
:label: eq:strat_update
y(t + \Delta t) \approx y(t) + f(y(t), t, \theta) \Delta t + g(y(t), t, \theta) \Delta W_t
```

Following the same chain rule logic as in @eq:chain_rule, the sensitivity of the loss with respect to the state at time $t$ is:

```{math}
:label: eq:strat_adjoint_step
a(t) = \left( \frac{\partial y(t+\Delta t)}{\partial y(t)} \right)^T a(t+\Delta t)
```

Substituting the derivative of @eq:strat_update:

```{math}
:label: eq:strat_adjoint_subst
a(t) \approx \left( I + \frac{\partial f}{\partial y}^T \Delta t + \sum_j \frac{\partial g_j}{\partial y}^T \Delta W_{t,j} \right) a(t+\Delta t)
```

In the limit $\Delta t \to 0$, this yields the adjoint SDE in Stratonovich form:

```{math}
:label: eq:strat_adjoint_sde
da(t) = -\left( \frac{\partial f}{\partial y} \right)^T a(t) dt - \sum_j \left( \frac{\partial g_j}{\partial y} \right)^T a(t) \circ dW_{t,j}
```

When converted back to Itô form for numerical stability and implementation, this introduces the **Stratonovich-to-Itô correction** term (see Appendix for details on stochastic calculus and the conversion formula):

```{math}
:label: eq:sde_adjoint
da(t) = -\left[ a(t) \frac{\partial f}{\partial y} - \sum_j \left( \frac{\partial g_j}{\partial y} \right) \left( a(t) \frac{\partial g_j}{\partial y} \right)^T \right] dt - \sum_j \left( a(t) \frac{\partial g_j}{\partial y} \right) dW_t
```

where $g_j$ are the columns of the diffusion matrix $g$. This allows NODEFit to learn diffusion processes with high precision and stability while maintaining a manageable memory footprint.

### Installation

NODEFit can be installed via pip:

```bash
pip install nodefit
```

### Sample Code

The following example demonstrates how to use NODEFit to fit a Neural ODE and a Neural SDE to synthetic time-series data.

```python
import numpy as np
import torch
import torch.nn as nn
from nodefit.neural_ode import NeuralODE
from nodefit.neural_sde import NeuralSDE

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# 1. Neural ODE Example
# Generate smooth synthetic time-series data (2D)
t = np.linspace(0, 5, 50)
y1 = 1.0 + 2.2 * (1 - np.exp(-0.5 * t))
y2 = 1.0 + 0.6 * (1 - np.exp(-0.5 * t))
data = np.stack([y1, y2], axis=1)

# Define the drift network (f_theta)
# Input: (t, y), so input_dim = 1 + 2 = 3
drift_nn = nn.Sequential(
    nn.Linear(3, 20),
    nn.Tanh(),
    nn.Linear(20, 2)
).double()

# Initialize and train the ODE model
ode = NeuralODE(drift_nn, t, data)
ode.train(num_epochs=500, print_every=100)

# 2. Neural SDE Example
# Generate noisy synthetic time-series data (2D)
y1_noisy = y1 + np.random.normal(0, 0.05, 50)
y2_noisy = y2 + np.random.normal(0, 0.05, 50)
data_noisy = np.stack([y1_noisy, y2_noisy], axis=1)

# Define the diffusion network (g_theta)
diffusion_nn = nn.Sequential(
    nn.Linear(3, 20),
    nn.Tanh(),
    nn.Linear(20, 2)
).double()

# Initialize and train the SDE model
sde = NeuralSDE(drift_nn, diffusion_nn, t, data_noisy, batch_size=10)
sde.train(num_epochs=300, print_every=10)
```

## Results

We evaluated NODEFit on both deterministic and stochastic benchmarks. The models were trained using the Adam optimizer with default learning rates.

### Neural ODE Results

For the deterministic case, we trained a Neural ODE for 500 epochs on a 2D system. As shown in @fig:ode_results, the model accurately captures the exponential growth and saturation of the system, providing smooth interpolations and stable extrapolations beyond the training range ($t > 5$).

:::{figure} results/ode_results.png
:label: fig:ode_results
Neural ODE fit and extrapolation. The dots represent training data, solid lines show the learned dynamics, and dashed lines represent extrapolation.
:::

### Neural SDE Results

In the stochastic case, we trained a Neural SDE for 300 epochs. The diffusion network learns to capture the noise characteristics of the data. @fig:sde_results illustrates the mean prediction along with the standard deviation (shaded regions) across multiple trajectories. The model successfully captures the underlying trend while quantifying the uncertainty, which increases during extrapolation.

:::{figure} results/sde_results.png
:label: fig:sde_results
Neural SDE fit and extrapolation. The shaded regions represent the standard deviation across 10 trajectories, capturing the learned diffusion process.
:::

## Conclusion

NODEFit offers a user-friendly and powerful tool for fitting continuous-time models to time-series data. By leveraging Neural ODEs and SDEs, it enables the discovery of governing laws from observations, bridging the gap between machine learning and physical modeling.

## Appendix: Stochastic Calculus and the Adjoint Method

The derivation of the stochastic adjoint sensitivity method relies on the choice of stochastic integral. This appendix provides the necessary background on the Itô and Stratonovich formulations.

### Martingales

A stochastic process $M_t$ is a **martingale** if its expected future value, given all the information available up to the current time $t$, is exactly its current value:

```{math}
:label: eq:martingale
E[M_s | \text{information up to time } t] = M_t
```

Intuitively, a martingale represents a "fair game" where there is no systematic tendency to increase or decrease. This property is fundamental for ensuring that a stochastic model does not have an unintended "hidden" drift.

### Itô vs. Stratonovich Integrals

For a stochastic process $y(t)$ governed by a diffusion term $g(y, t)$, the integral with respect to Brownian motion $W_t$ can be defined in two primary ways depending on the evaluation point within a time interval $[t, t+\Delta t]$:

1. **Itô Integral** (denoted $g \, dW_t$): Evaluates the integrand at the **left endpoint** $t$. Because the integrand is evaluated before the noise increment $dW_t$ occurs, they are independent. Since Brownian increments have zero mean, the expected change is zero, making the Itô integral a **martingale**. This makes it the standard choice for modeling physical systems where noise should not introduce systematic drift. However, it does not follow the standard chain rule of calculus.
2. **Stratonovich Integral** (denoted $g \circ dW_t$): Evaluates the integrand at the **midpoint** $t + \Delta t/2$. This creates a correlation between the integrand and the noise, which introduces a drift and causes the integral to **lose the martingale property**. However, its primary advantage is that it **obeys the standard rules of calculus** (chain rule, product rule), which simplifies the derivation of adjoint equations.

### Conversion and Correction

The relationship between a Stratonovich SDE ($dy = f_s dt + g \circ dW_t$) and an Itô SDE ($dy = f_i dt + g dW_t$) is given by the conversion formula:

```{math}
:label: eq:ito_strat_conv
f_i(y, t) = f_s(y, t) + \frac{1}{2} \sum_j \left( \frac{\partial g_j(y, t)}{\partial y} \right) g_j(y, t)
```

where $g_j$ are the columns of the diffusion matrix. The second term is the **Stratonovich-to-Itô correction**. In `torchsde`, derivations are performed in the Stratonovich framework to leverage standard calculus, while numerical solvers often operate in the Itô framework, requiring the explicit inclusion of this correction term in the drift dynamics.

## References

