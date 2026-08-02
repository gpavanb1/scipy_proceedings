---
title: "NODEFit: Fit time-series data with a Neural Differential Equation"
abstract: |
  Time-series data often arise from underlying physical, biological, and engineering systems governed by continuous dynamics. Standard discrete-time regressors and fixed-form curve fits can interpolate observations yet fail to extrapolate or to represent stochastic structure when the governing equations are unknown. Neural Ordinary Differential Equations (Neural ODEs) and Neural Stochastic Differential Equations (Neural SDEs) learn evolution laws directly from data. This paper introduces NODEFit, an open-source Python package that provides a simple and efficient interface for fitting ODEs and SDEs to measured data. We motivate the approach with examples where conventional methods do not perform well, demonstrate recovery of underlying dynamics on noisy benchmarks, and summarize when practitioners should prefer continuous-time neural models.
---

## Introduction

Physical phenomena are commonly governed by differential equations. Traditional methods for time-series analysis often rely on discrete-time models, which may fail to capture the underlying continuous-time dynamics. Neural Ordinary Differential Equations (Neural ODEs) [@chen2018torchdiffeq] represent a paradigm shift by modeling the latent state evolution as a continuous process.

Consider a concrete motivating example where two state variables rise toward distinct steady-state values according to smooth exponential kinetics, and each measurement is corrupted by noise. One might try nonlinear least squares with `scipy.optimize.curve_fit`, specifying a candidate functional form such as a single exponential or logistic. That approach works only when the chosen template matches the true dynamics. 

These failures share a root cause: the models describe values at sampled times rather than information of rates of change that generated the trajectory. Neural Ordinary Differential Equations (Neural ODEs) [@chen2018torchdiffeq] and Neural Stochastic Differential Equations (Neural SDEs) [@li2020scalable; @kidger2021neural] take the opposite view. Instead of fitting $y(t)$ directly, they learn a vector field $f_\theta(y, t)$—and, when needed, a diffusion term $g_\theta(y, t)$—such that integrating forward reproduces the observations. When noise is intrinsic rather than measurement error, a Neural SDE separates drift from diffusion, yielding mean trajectories and uncertainty bands that widen naturally outside the data.

NODEFit packages these ideas for practitioners. It provides a simple Python interface for fitting ODEs and SDEs to measured data, wrapping differentiable solvers and adjoint-based training so that users can focus on specifying a compact neural network for the dynamics rather than on solver internals. Benchmarks in this paper show that this continuous-time formulation recovers the underlying kinetics and extrapolates reliably on the above mentioned motivating example, which polynomial, template-based, and discrete-time alternatives struggle to match without prior knowledge of the governing equations.

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

NODEFit is implemented as an open-source Python package built on top of the PyTorch ecosystem. It leverages specialized libraries to handle the numerical integration and gradient computation required for training Neural ODEs and SDEs. By abstracting these complexities, NODEFit makes it remarkably easy to fit complex time-series data to governing differential equations. All plots in this paper were generated using Matplotlib [@matplotlib].

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

This enables the training of complex models on large time-series datasets that would otherwise be computationally prohibitive. The memory efficiency stems from the fact that the adjoint method does not require storing intermediate states $y(t)$ from the forward pass. Instead, the original ODE is solved backwards in time alongside the adjoint equation, reconstructing the state $y(t)$ on the fly. This reduces the memory complexity from $O(N)$, where $N$ is the number of solver steps, to $O(1)$ relative to the trajectory length.

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
da(t) = -\left[ a(t) \frac{\partial f}{\partial y} - \sum_j \left( a(t) \frac{\partial g_j}{\partial y} \right) \frac{\partial g_j}{\partial y} \right] dt - \sum_j \left( a(t) \frac{\partial g_j}{\partial y} \right) dW_t
```

where $g_j$ are the columns of the diffusion matrix $g$. Note that while the standard Stratonovich-to-Itô conversion for a forward SDE includes a $1/2$ factor, this factor is absorbed in the adjoint case because the correction term for the adjoint state $a(t)$ involves two symmetric components that sum together, as derived in @li2020scalable. This allows NODEFit to learn diffusion processes with high precision and stability while maintaining a manageable memory footprint. Similar to the deterministic case, the stochastic adjoint avoids storing the full trajectory. However, SDEs require consistent noise across both forward and backward passes. `torchsde` achieves this through a **Virtual Brownian Tree**, which allows for the exact reconstruction of the Brownian motion $W_t$ at any time point using a fixed seed. By reconstructing both the state and the noise during the backward pass, the memory cost remains constant even for complex stochastic trajectories.

#### Memory Efficiency Comparison

The primary advantage of the adjoint methods used in NODEFit is the reduction in memory overhead. The following table summarizes the comparison between the naive backpropagation approach and the adjoint-based methods implemented in `torchdiffeq` and `torchsde`.

:::{table} Comparison of memory efficiency and computational trade-offs between naive backpropagation and the adjoint sensitivity method.
:label: table:memory_comparison

| Feature | Naive Backprop | Adjoint Method (NODEFit) |
| :--- | :--- | :--- |
| **Intermediate States** | Stored in memory | Reconstructed on the fly |
| **Memory Scaling** | $O(N)$ (Linear with steps) | $O(1)$ (Constant with steps) |
| **Noise (SDEs)** | Must be stored for every step | Regenerated via Virtual Brownian Tree |
| **Trade-off** | Faster (no reconstruction) | Slower (requires solving backwards) |
:::

### Performance Optimizations

To handle larger datasets and more complex trajectories, we utilized an optimized implementation that inherits from the base `NeuralSDE` and `SDE` classes. This version leverages faster tensor operations for state-time concatenation and an efficient training loop. For the results presented in this paper, we tuned the following hyperparameters: a `batch_size` of 20 for improved gradient estimates and a fixed time step `dt` of 0.1 to balance numerical stability with computational speed.

### Installation

NODEFit can be installed via pip:

```bash
pip install nodefit
```

### Sample Code

The following example demonstrates how to use NODEFit to fit a Neural SDE to a 2D trajectory with noise, showcasing how the high-level API makes it remarkably easy to fit complex time-series data.

```python
import numpy as np
import torch
import torch.nn as nn
from nodefit.neural_sde import NeuralSDE

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Generate noisy synthetic time-series data (2D)
t = np.linspace(0, 5, 50)
y1 = 1.0 + 2.2 * (1 - np.exp(-0.5 * t)) + np.random.normal(0, 0.1, 50)
y2 = 1.0 + 0.6 * (1 - np.exp(-0.5 * t)) + np.random.normal(0, 0.1, 50)
data = np.stack([y1, y2], axis=1)

# Define the drift network (f_theta)
# Input: (t, y), so input_dim = 1 + 2 = 3
drift_nn = nn.Sequential(
    nn.Linear(3, 20),
    nn.Tanh(),
    nn.Linear(20, 2)
).double()

# Define the diffusion network (g_theta)
diffusion_nn = nn.Sequential(
    nn.Linear(3, 20),
    nn.Tanh(),
    nn.Linear(20, 2)
).double()

# Initialize and train the SDE model: fitting data is as simple as passing the observations
sde = NeuralSDE(drift_nn, diffusion_nn, t, data, batch_size=20)
sde.train(num_epochs=500, print_every=100)

# Extrapolate to future time points
extrapolated = sde.extrapolate(tf=8, npts=40)
```

## Results
(sec:results)=

We evaluated NODEFit on both deterministic and stochastic benchmarks drawn from the motivating scenario in the Introduction: coupled states approaching saturation with optional noise. The models were trained using the Adam optimizer with default learning rates. In each case, the goal is not merely to interpolate scattered points but to recover a coherent evolution law that extrapolates beyond $t = 5$. @fig:ode_results through @fig:trajectory_results illustrate settings where conventional curve fitting would require the correct functional template *a priori*, and where treating noise as homoscedastic regression error would misrepresent uncertainty during forecast.

### Neural ODE Results

For the deterministic case, we first fit the same training data with `scipy.optimize.curve_fit`, using a cubic polynomial ($y = a + bt + ct^2 + dt^3$) fitted independently to each state. The extra flexibility tracks the training window closely, but without a saturation mechanism the extrapolation past $t = 5$ inflects upward rather than leveling off (@fig:ode_results, dashed curves). We then trained a Neural ODE for 1000 epochs on the same 2D system. The learned flow matches the saturating trajectories and continues smoothly toward steady state beyond the training window, without specifying the functional form in advance.

:::{figure} results/ode_results.png
:label: fig:ode_results
Baseline failure and Neural ODE success on coupled saturating kinetics. Dashed curves show `curve_fit` with a cubic polynomial template; solid curves show the Neural ODE integrated through $t = 10$. The vertical dotted line marks the end of training data ($t = 5$).
:::

### Neural SDE Results

In the stochastic case, we trained a Neural SDE for 300 epochs. The diffusion network learns to capture the noise characteristics of the data. @fig:sde_results illustrates the mean prediction along with the standard deviation (shaded regions) across multiple trajectories. The model successfully captures the underlying trend while quantifying the uncertainty, which increases during extrapolation.

:::{figure} results/simple_sde_results.png
:label: fig:sde_results
Neural SDE fit and extrapolation. The shaded regions represent the standard deviation across 10 trajectories, capturing the learned diffusion process.
:::

### Fitting Complex Dynamics

We also tested NODEFit on a more complex 2D system with noise. The underlying theoretical trajectories for this system are governed by the following equations:

```{math}
:label: eq:theoretical_mean
\begin{aligned}
y_1(t) &= 1.0 + 2.2(1 - e^{-0.5t}) \\
y_2(t) &= 1.0 + 0.6(1 - e^{-0.5t})
\end{aligned}
```

The model was tasked with learning these underlying dynamics and providing robust extrapolations. As shown in @fig:trajectory_results, the Neural SDE successfully recovers the mean trajectory, closely matching the underlying theoretical dynamics even in the presence of noise. The comparison between the theoretical mean and the predicted mean demonstrates the model's ability to filter out stochastic fluctuations and capture the true governing laws.

:::{figure} results/trajectory_sde_results.png
:label: fig:trajectory_results
Neural SDE fit on a complex 2D trajectory. The dotted black line represents the underlying theoretical trajectory, while the solid lines and shaded regions show the predicted mean and uncertainty. The model captures the multi-dimensional dynamics and provides reliable extrapolations.
:::

## Conclusion

NODEFit offers a user-friendly and powerful tool for fitting continuous-time models to time-series data. By leveraging Neural ODEs and SDEs, it enables the discovery of governing laws from observations, bridging the gap between machine learning and physical modeling.

For practitioners deciding whether to use NODEFit, the central question is whether the data plausibly arise from a smooth, Markovian continuous-time process. If polynomial regression, splines, or `scipy.optimize.curve_fit` with a hand-chosen template already produce stable fits and credible extrapolations, a Neural ODE may be unnecessary. Consider NODEFit when those standard tools leave systematic residuals, extrapolations diverge from physical expectations, or the functional form of the dynamics is unknown. Template-based fits must guess a separate closed-form expression for each channel, whereas a Neural ODE learns a single vector field coupling all states.

When observations are noisy, ask whether the noise reflects measurement error alone or variability intrinsic to the process. Ordinary least squares and deterministic Neural ODEs treat scatter as something to be averaged out. If uncertainty grows with state magnitude or if extrapolated forecasts should carry confidence intervals, a Neural SDE is the more appropriate NODEFit model. The diffusion network learns state-dependent noise alongside the drift.

Network capacity deserves equal attention. The drift and diffusion networks should be the smallest architectures that fit the underlying process. When domain knowledge suggests a low-dimensional manifold or saturating kinetics, prefer shallow networks with smooth activations, as in the examples here. When the mechanism is unknown, treat width and depth as hyperparameters to be validated on held-out time intervals or early/late segments of the series as over-parameterized networks can interpolate noise yet fail to extrapolate. NODEFit's API makes tuning actions easy to perform, such as, swap network sizes, retrain, and compare extrapolations on the same solver settings.

## CRediT authorship contribution statement

**Pavan B. Govindaraju**: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing – original draft, Writing – review & editing.

Portions of this work were assisted using generative AI tool - Cursor. The tools were used for drafting text, refining language, or generating code suggestions. All outputs were reviewed, verified, and revised by the author(s), who take full responsibility for the accuracy and integrity of the final content.

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

It is important to note that for the adjoint SDE (Equation {ref}`eq:sde_adjoint`), the $1/2$ factor is absent. This is because the correction term for the adjoint state $a(t)$ arises from the interaction between the forward state $y(t)$ and the adjoint variable. When converting the augmented system $(y, a)$ to Itô form, the resulting drift correction for $a(t)$ consists of two identical terms from the Stratonovich expansion that sum to unity, effectively canceling the $1/2$ coefficient found in the standard forward conversion formula [@li2020scalable].

#### Mathematical Derivation

To see this mathematically, we use index notation where $y_i$ and $a_i$ denote the components of the state and adjoint (row) vectors. Consider a forward Itô SDE $dy_i = f_i dt + \sum_j g_{i,j} dW_j$. The equivalent Stratonovich drift $\tilde{f}_i$ is:

```{math}
:label: eq:strat_drift_index
\tilde{f}_i = f_i - \frac{1}{2} \sum_j \sum_k \frac{\partial g_{i,j}}{\partial y_k} g_{k,j}
```

The adjoint Stratonovich SDE is $da_i = - \sum_k a_k \frac{\partial \tilde{f}_k}{\partial y_i} dt - \sum_j \sum_k a_k \frac{\partial g_{k,j}}{\partial y_i} \circ dW_j$. To convert this back to Itô form, we identify the adjoint diffusion term $\sigma_{i,j}^{(a)} = - \sum_k a_k \frac{\partial g_{k,j}}{\partial y_i}$. The Itô drift correction $C_{a_i}$ for the adjoint is:

```{math}
:label: eq:ito_corr_index
C_{a_i} = \frac{1}{2} \sum_j \left( \sum_k \frac{\partial \sigma_{i,j}^{(a)}}{\partial a_k} \sigma_{k,j}^{(a)} + \sum_k \frac{\partial \sigma_{i,j}^{(a)}}{\partial y_k} g_{k,j} \right)
```

Expanding these terms:
1. $\sum_k \frac{\partial \sigma_{i,j}^{(a)}}{\partial a_k} \sigma_{k,j}^{(a)} = \sum_k \left( -\frac{\partial g_{k,j}}{\partial y_i} \right) \left( -\sum_m a_m \frac{\partial g_{m,j}}{\partial y_k} \right) = \sum_m a_m \sum_k \frac{\partial g_{m,j}}{\partial y_k} \frac{\partial g_{k,j}}{\partial y_i}$
2. $\sum_k \frac{\partial \sigma_{i,j}^{(a)}}{\partial y_k} g_{k,j} = \sum_k \left( -\sum_m a_m \frac{\partial^2 g_{m,j}}{\partial y_k \partial y_i} \right) g_{k,j} = - \sum_m a_m \sum_k \frac{\partial^2 g_{m,j}}{\partial y_i \partial y_k} g_{k,j}$

Now, we differentiate the Stratonovich drift $\tilde{f}_k$ from @eq:strat_drift_index:
$\frac{\partial \tilde{f}_k}{\partial y_i} = \frac{\partial f_k}{\partial y_i} - \frac{1}{2} \sum_j \sum_m \left( \frac{\partial^2 g_{k,j}}{\partial y_i \partial y_m} g_{m,j} + \frac{\partial g_{k,j}}{\partial y_m} \frac{\partial g_{m,j}}{\partial y_i} \right)$.

Combining everything into the total Itô drift for $a_i$:
$\mu_{a_i} = - \sum_k a_k \frac{\partial \tilde{f}_k}{\partial y_i} + C_{a_i}$
$\mu_{a_i} = - \sum_k a_k \left[ \frac{\partial f_k}{\partial y_i} - \frac{1}{2} \sum_j \sum_m \left( \frac{\partial^2 g_{k,j}}{\partial y_i \partial y_m} g_{m,j} + \frac{\partial g_{k,j}}{\partial y_m} \frac{\partial g_{m,j}}{\partial y_i} \right) \right] + \frac{1}{2} \sum_j \sum_m a_m \left( \sum_k \frac{\partial g_{m,j}}{\partial y_k} \frac{\partial g_{k,j}}{\partial y_i} - \sum_k \frac{\partial^2 g_{m,j}}{\partial y_i \partial y_k} g_{k,j} \right)$

The terms involving second derivatives $\frac{\partial^2 g}{\partial y^2}$ cancel out, and the terms involving the product of first derivatives $\frac{\partial g}{\partial y} \frac{\partial g}{\partial y}$ add up: $\frac{1}{2} + \frac{1}{2} = 1$. This yields the final Itô drift:
$\mu_{a_i} = - \sum_k a_k \frac{\partial f_k}{\partial y_i} + \sum_j \sum_k a_k \sum_m \frac{\partial g_{k,j}}{\partial y_m} \frac{\partial g_{m,j}}{\partial y_i}$, which is the component-wise form of Equation {ref}`eq:sde_adjoint`.
