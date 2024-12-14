import pytest
import os
import numpy as np
from unittest.mock import mock_open, patch
from tools.ipsc_classifier_vgg.plot_training_metrics import (
    read_scores,
    read_metrics,
    calculate_confidence_intervals,
    plot_scores_with_ci,
    plot_metrics
)


# Test the read_scores function
def test_read_scores():
    """
    Tests the read_scores function to ensure it correctly reads numerical scores from a file.
    It mocks a file with sample data and checks if the returned list matches the expected scores.
    """
    mock_data = "0.9\n0.8\n0.85\n0.95\n"
    
    with patch("builtins.open", mock_open(read_data=mock_data)):
        scores = read_scores("mock_file.txt")
    
    expected_scores = [0.9, 0.8, 0.85, 0.95]
    assert scores == expected_scores


# Test the read_metrics function
def test_read_metrics():
    """
    Tests the read_metrics function to ensure it correctly reads accuracy, precision, recall, and F1 scores
    from a file. The file content is mocked, and the function should return four lists corresponding to these metrics.
    """
    mock_data = "0.9 0.85 0.8 0.75\n0.91 0.86 0.81 0.77\n0.92 0.87 0.82 0.78\n"
    
    with patch("builtins.open", mock_open(read_data=mock_data)):
        accuracy, precision, recall, F1 = read_metrics("mock_metrics.txt")
    
    expected_accuracy = [0.9, 0.91, 0.92]
    expected_precision = [0.85, 0.86, 0.87]
    expected_recall = [0.8, 0.81, 0.82]
    expected_F1 = [0.75, 0.77, 0.78]
    
    assert accuracy == expected_accuracy
    assert precision == expected_precision
    assert recall == expected_recall
    assert F1 == expected_F1


# Test the calculate_confidence_intervals function
def test_calculate_confidence_intervals():
    """
    Tests the calculate_confidence_intervals function. It verifies that the confidence intervals
    are correctly calculated for a set of scores based on the provided mean, standard deviation, and factor.
    """
    scores = [0.9, 0.8, 0.85, 0.95]
    mean = 0.875
    std = 0.058
    factor = 0.1
    
    ci = calculate_confidence_intervals(scores, mean, std, factor)
    
    expected_ci = [
        0.1 * score * std / mean for score in scores
    ]
    
    assert np.allclose(ci, expected_ci, atol=1e-2)


# Test the plot_scores_with_ci function (mocking plt)
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_plot_scores_with_ci(mock_show, mock_savefig):
    """
    Tests the plot_scores_with_ci function. This test mocks the `show` and `savefig` functions from `matplotlib.pyplot`
    to ensure that the plot is displayed correctly without actually saving the plot to a file.
    """
    t = np.array([1, 2, 3])
    y_train = [0.8, 0.9, 0.85]
    y_valid = [0.85, 0.88, 0.9]
    ci_train = [0.01, 0.02, 0.015]
    ci_valid = [0.02, 0.01, 0.025]
    
    plot_scores_with_ci(t, y_train, y_valid, ci_train, ci_valid, save_dir=None)
    
    # Ensure the plot is shown
    mock_show.assert_called_once()

    # Ensure savefig is not called because save_dir is None
    mock_savefig.assert_not_called()


# Test the plot_metrics function (mocking plt)
@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_plot_metrics(mock_show, mock_savefig):
    """
    Tests the plot_metrics function. This test mocks the `show` and `savefig` functions from `matplotlib.pyplot`
    to verify that the plot is displayed correctly and not saved to a file when `save_dir` is None.
    """
    t = np.array([1, 2, 3])
    accuracy = [0.9, 0.91, 0.92]
    precision = [0.85, 0.86, 0.87]
    recall = [0.8, 0.81, 0.82]
    F1 = [0.75, 0.77, 0.78]
    
    plot_metrics(t, accuracy, precision, recall, F1, save_dir=None)
    
    # Ensure the plot is shown
    mock_show.assert_called_once()

    # Ensure savefig is not called because save_dir is None
    mock_savefig.assert_not_called()


if __name__ == "__main__":
    pytest.main()