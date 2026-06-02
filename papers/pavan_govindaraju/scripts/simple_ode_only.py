import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nodefit.neural_ode import NeuralODE

# Set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def run_ode_example():
    print("Running Neural ODE example...")
    # Generate smooth synthetic time-series data (2D)
    t = np.linspace(0, 5, 50)
    y1 = 1.0 + 2.2 * (1 - np.exp(-0.5 * t))
    y2 = 1.0 + 0.6 * (1 - np.exp(-0.5 * t))
    data = np.stack([y1, y2], axis=1)

    # Define the drift network (f_theta)
    # The input to the drift network is (t, y), so input_dim = 1 + data_dim = 1 + 2 = 3
    drift_nn = nn.Sequential(
        nn.Linear(3, 20),
        nn.Tanh(),
        nn.Linear(20, 2)
    ).double()

    # Initialize and train the model
    ode = NeuralODE(drift_nn, t, data)
    ode.train(num_epochs=500, print_every=100)

    # Extrapolate beyond the training range
    extrapolated = ode.extrapolate(tf=10, npts=50)
    
    # Plot and save
    ode.plot(extra_data=extrapolated)
    plt.savefig('ode_results.png')
    plt.close()
    print("Neural ODE example finished. Plot saved as ode_results.png")

if __name__ == "__main__":
    run_ode_example()
