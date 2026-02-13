import numpy as np
import matplotlib.pyplot as plt

def simulate_random_walks(n_steps, n_walks, seed = None):
    """
    Simulate 1-D symmetric random walks.

    Parameters
    ----------
    n_steps : Number of steps in the walk
    n_walks : Number of walks completed
    seed : Random seed for reproducibility

    Returns
    ----------
    An array of shape (n_walks, n_steps) containing step increments

    """
    if seed is not None:
        np.random.seed(seed)

    draws = np.random.choice([-1,1], size=(n_walks, n_steps))

    return draws

def cumulative_positions(steps):
    """
    Converts steps increments into cumulative positions

    Parameters
    ----------
    steps : An array of shape (n_walks, n_steps)

    Returns
    ----------
    An array of shape (n_walks, n_steps+1) containing positions

    """
    start_position = np.zeros([steps.shape[0], 1])
    positions = np.append(start_position, steps.cumsum(axis=1), axis=1)
    return positions

def plot_walks(positions, max_walks=20):
    """
    Plots sample random walk paths

    Parameters
    ----------
    positions : Cumulative positions array of shape (n_walks, n_steps+1)
    max_walks : Maximum number of walks to plot

    """
    n_plot = min(max_walks, positions.shape[0])
    plt.plot(positions[:n_plot].T)
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.title("Sample Random Walk Paths")
    plt.show()

def plot_final_positions(positions):
    """
    Plots histogram of final positions

    Parameters
    ----------
    positions : Cumulative positions array of shape (n_walks, n_steps+1)
    """
    final_positions = positions[:,-1]
    plt.hist(final_positions, bins=30, density=True)
    plt.xlabel("Final position")
    plt.ylabel("Density")
    plt.title("Distribution of Final Positions")
    plt.show()


def plot_variance_growth(positions):
    """
    Plots variance of positions as a function of time

    Parameters
    ----------
    positions : Cumulative positions array of shape (n_walks, n_steps+1)
    """
    variance_growth = np.var(positions, axis=0)
    plt.plot(variance_growth)
    plt.xlabel("Time step")
    plt.ylabel("Variance")
    plt.title("Variance Growth of Random Walks")
    plt.show()

if __name__ == "__main__":
    steps = simulate_random_walks(n_steps=1_000, n_walks=5_000, seed=42)
    positions = cumulative_positions(steps)

    plot_walks(positions)
    plot_final_positions(positions)
    plot_variance_growth(positions)