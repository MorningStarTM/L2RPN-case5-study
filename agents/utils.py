import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def plot_learning(history, filename, window=50):
    """
    Plot the learning curve.

    Args:
        history: The list of scores.
        filename: The name of the file to save the plot.
        window: The window size for the moving average.
    """
    N = len(history)
    if N == 0:
        print("Empty history provided. No plot will be generated.")
        return
    
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(history[max(0, t - window):(t + 1)])
    
    plt.plot(running_avg)
    plt.title('Running average of previous {} scores'.format(window))
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.savefig(filename)
    plt.show()


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)




def plotLearning(x, scores, filename, window=100):
    """
    Plots the scores and the running average of the scores.

    Args:
        x (list): The episode numbers.
        scores (list): The scores achieved in each episode.
        filename (str): The filename to save the plot as.
        window (int): The window size for calculating the running average.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot scores
    ax.plot(x, scores, label='Scores', color='blue')

    # Calculate and plot running average of scores
    if len(scores) >= window:
        running_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax.plot(x[:len(running_avg)], running_avg, label=f'Running Average (window={window})', color='red')

    # Labels and legends
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Scores', color='blue')
    ax.tick_params(axis='y', colors='blue')
    ax.legend(loc='upper left')

    # Save plot
    plt.title('Training Progress')
    plt.savefig(filename)
    plt.close()




def print_progress_bar(iteration, total, length=50):
    """
    Prints a progress bar with `#` symbols.
    
    Args:
    - iteration: Current iteration (int)
    - total: Total iterations (int)
    - length: Character length of bar (default is 50)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '#' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent}% Complete')
    sys.stdout.flush()