import sys
import os
sys.path.append('pybrain/pybrain')

import pandas as pd
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_prepare_datasets(csv_file='dataset03.csv', test_size=0.2, random_state=42):
    """
    Load CSV file and create PyBrain training and testing datasets
    
    Parameters:
    csv_file (str): Path to the CSV file
    test_size (float): Proportion of dataset to include in test split (0.0 to 1.0)
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (train_dataset, test_dataset, input_size, target_size)
    """
    
    # Load the CSV file
    print("Loading dataset...")
    data = pd.read_csv(csv_file)
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Extract features (x) and target (y)
    # Assuming 'x' is the input feature and 'y' is the target variable
    X = data['x'].values.reshape(-1, 1)  # Reshape to 2D array
    y = data['y'].values.reshape(-1, 1)  # Reshape to 2D array
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create PyBrain datasets
    input_size = 1  # Since we have one input feature 'x'
    target_size = 1  # Since we have one target variable 'y'
    
    # Create supervised dataset structure
    train_dataset = SupervisedDataSet(input_size, target_size)
    test_dataset = SupervisedDataSet(input_size, target_size)
    
    # Add samples to training dataset
    for i in range(len(X_train)):
        train_dataset.addSample(X_train[i], y_train[i])
    
    # Add samples to testing dataset
    for i in range(len(X_test)):
        test_dataset.addSample(X_test[i], y_test[i])
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset, input_size, target_size

def create_and_train_model(train_dataset, test_dataset, input_size, target_size, hidden_neurons=5, epochs=100):
    """
    Create and train a neural network model using PyBrain
    
    Parameters:
    train_dataset: PyBrain training dataset
    test_dataset: PyBrain testing dataset
    input_size (int): Number of input features
    target_size (int): Number of output targets
    hidden_neurons (int): Number of neurons in hidden layer
    epochs (int): Number of training epochs
    
    Returns:
    tuple: (trained_network, trainer)
    """
    
    # Build the neural network
    print("\nBuilding neural network...")
    network = buildNetwork(input_size, hidden_neurons, target_size, bias=True)
    
    # Create trainer
    trainer = BackpropTrainer(network, train_dataset, learningrate=0.01, verbose=True)
    
    # Train the network
    print(f"Training the model for {epochs} epochs...")
    training_errors = []
    for i in range(epochs):
        error = trainer.train()
        training_errors.append(error)
        if i % 20 == 0:
            print(f"Epoch {i}: Error = {error:.6f}")
    
    # Test the model
    print("\nTesting the model...")
    test_error = trainer.testOnData(test_dataset)
    print(f"Test error: {test_error:.6f}")
    
    return network, trainer, training_errors

# Main execution
if __name__ == "__main__":
    try:
        # Load and prepare datasets
        train_ds, test_ds, input_dim, target_dim = load_and_prepare_datasets('dataset03.csv')
        
        # Create and train model
        trained_network, trainer, errors = create_and_train_model(
            train_ds, test_ds, input_dim, target_dim, 
            hidden_neurons=5, epochs=100
        )
        
        print("\nModel training completed successfully!")
        print(f"Final training error: {errors[-1]:.6f}")
        
        # Example: Make a prediction
        if len(test_ds) > 0:
            sample_input = test_ds['input'][0]
            sample_target = test_ds['target'][0]
            prediction = trained_network.activate(sample_input)
            print(f"\nSample prediction:")
            print(f"Input: {sample_input[0]:.4f}, Target: {sample_target[0]:.4f}, Prediction: {prediction[0]:.4f}")
            
    except FileNotFoundError:
        print("Error: 'dataset03.csv' file not found. Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

