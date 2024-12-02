import numpy as np
import os
import matplotlib.pyplot as plt

def read_scores(file_path):
    """
    Reads a text file containing numerical scores and returns them as a list of floats.
    
    Args:
        file_path (str): Path to the text file containing numerical scores.
        
    Returns:
        list: List of floats representing the scores from the file.
    """
    scores = []
    with open(file_path, "r") as file:
        for line in file:
            scores.append(float(line))
    return scores

def read_metrics(file_path):
    """
    Reads a text file containing metrics (accuracy, precision, recall, F1) for each epoch.
    
    Args:
        file_path (str): Path to the text file containing the metrics for each epoch.
        
    Returns:
        tuple: Four lists representing accuracy, precision, recall, and F1 scores for each epoch.
    """
    accuracy, precision, recall, F1 = [], [], [], []
    with open(file_path, "r") as file:
        for line in file:
            lst = line.split()
            accuracy.append(float(lst[0]))
            precision.append(float(lst[1]))
            recall.append(float(lst[2]))
            F1.append(float(lst[3]))
    return accuracy, precision, recall, F1

def calculate_confidence_intervals(scores, mean, std, factor=0.1):
    """
    Calculates the confidence intervals for a given set of scores.
    
    Args:
        scores (list): List of numerical scores.
        mean (float): Mean of the scores.
        std (float): Standard deviation of the scores.
        factor (float): Multiplier factor to calculate the confidence interval. Default is 0.1.
        
    Returns:
        list: List of confidence intervals for the scores.
    """
    ci = []
    for score in scores:
        ci.append(factor * score * std / mean)
    return ci

def plot_scores_with_ci(t, y_train, y_valid, ci_train, ci_valid, save_dir="./metrics"):
    """
    Plots training and validation scores along with confidence intervals using matplotlib.
    
    Args:
        t (numpy.ndarray): Array of epochs.
        y_train (list): List of training scores.
        y_valid (list): List of validation scores.
        ci_train (list): List of confidence intervals for training scores.
        ci_valid (list): List of confidence intervals for validation scores.
        save_dir (str, optional): Directory to save the plot. If None, the plot will be shown but not saved.
        
    Returns:
        None: Displays and optionally saves the plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot validation and training scores as lines
    plt.plot(t, y_valid, 'b-o', label='Validation')
    plt.plot(t, y_train, 'r-o', label='Training')

    # Plot confidence intervals for validation
    y_u_valid = [y_valid[i] + ci_valid[i] for i in range(len(y_valid))]
    y_l_valid = [y_valid[i] - ci_valid[i] for i in range(len(y_valid))]
    y_l_valid = y_l_valid[::-1]  # Reverse for filling below
    plt.fill_between(np.concatenate([t, t[::-1]]), np.concatenate([y_u_valid, y_l_valid]), 
                     color='blue', alpha=0.2, label='Validation CI')

    # Plot confidence intervals for training
    y_u_train = [y_train[i] + ci_train[i] for i in range(len(y_train))]
    y_l_train = [y_train[i] - ci_train[i] for i in range(len(y_train))]
    y_l_train = y_l_train[::-1]  # Reverse for filling below
    plt.fill_between(np.concatenate([t, t[::-1]]), np.concatenate([y_u_train, y_l_train]), 
                     color='red', alpha=0.2, label='Training CI')

    # Labels and Title
    plt.title("Training and Validation Scores with Confidence Intervals")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend(title="Phase")

    if save_dir:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Create a path for the file
        file_path = os.path.join(save_dir, "training_validation_plot.png")
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    
    plt.show()

def plot_metrics(t, accuracy, precision, recall, F1, save_dir="./metrics"):
    """
    Plots accuracy, precision, recall, and F1 scores over epochs using matplotlib.
    
    Args:
        t (numpy.ndarray): Array of epochs.
        accuracy (list): List of accuracy scores for each epoch.
        precision (list): List of precision scores for each epoch.
        recall (list): List of recall scores for each epoch.
        F1 (list): List of F1 scores for each epoch.
        save_dir (str, optional): Directory to save the plot. If None, the plot will be shown but not saved.
        
    Returns:
        None: Displays and optionally saves the plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot each metric as a line
    plt.plot(t, accuracy, 'b-o', label='Accuracy')
    plt.plot(t, precision, 'g-o', label='Precision')
    plt.plot(t, recall, 'r-o', label='Recall')
    plt.plot(t, F1, 'c-o', label='F1')

    # Labels and Title
    plt.title("Metrics Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend(title="Metrics")

    if save_dir:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Create a path for the file
        file_path = os.path.join(save_dir, "Accuracy_F1_Recall_Precision_plot.png")
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")

    
    plt.show()

def main():
    # Set model directory and paths to required files
    my_model_directory = os.path.join('./', 'models')
    y_train = read_scores(os.path.join(my_model_directory, "epochs_train.txt"))
    y_valid = read_scores(os.path.join(my_model_directory, "epochs_valid.txt"))

    # Calculate confidence intervals
    valid_scores_mean = np.mean(y_valid)
    valid_scores_std = np.std(y_valid)
    train_scores_mean = np.mean(y_train)
    train_scores_std = np.std(y_train)

    ci_valid = calculate_confidence_intervals(y_valid, valid_scores_mean, valid_scores_std)
    ci_train = calculate_confidence_intervals(y_train, train_scores_mean, train_scores_std)

    # Read metrics for accuracy, precision, recall, and F1
    accuracy, precision, recall, F1 = read_metrics(os.path.join(my_model_directory, "epochs_valid_metrics.txt"))

    # Create time array for plotting
    t = np.linspace(1, len(y_valid), len(y_valid))

    # Plot training and validation scores with confidence intervals
    plot_scores_with_ci(t, y_train, y_valid, ci_train, ci_valid, save_dir="./metrics_plots")

    # Plot accuracy, precision, recall, and F1
    plot_metrics(t, accuracy, precision, recall, F1, save_dir="./metrics_plots")


if __name__ == "__main__":
    main()