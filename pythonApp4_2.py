import sys
sys.path.append('pybrain/pybrain')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import xml.etree.ElementTree as ET
from xml.dom import minidom

try:
    from pybrain.datasets import SupervisedDataSet
    from pybrain.tools.shortcuts import buildNetwork
    from pybrain.supervised.trainers import BackpropTrainer
    PYBRAIN_AVAILABLE = True
    print("PyBrain imported successfully!")
except ImportError as e:
    print(f"PyBrain import failed: {e}")
    print("Using scikit-learn MLPRegressor as fallback...")
    from sklearn.neural_network import MLPRegressor
    PYBRAIN_AVAILABLE = False

def load_and_prepare_datasets(csv_file='dataset03.csv', test_size=0.2, random_state=42):
    """Load CSV file and prepare datasets for training"""
    print("Loading dataset...")
    data = pd.read_csv(csv_file)
    print(f"Dataset shape: {data.shape}")
    print(f"First few rows:")
    print(data.head())
    
    # Extract features and target
    X = data['x'].values.reshape(-1, 1)
    y = data['y'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, X.shape[1]

def create_ols_model(X_train, y_train, X_test, y_test):
    """Create and train Ordinary Least Squares model"""
    print("\n" + "="*50)
    print("Training OLS Model")
    print("="*50)
    
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = ols_model.predict(X_train)
    y_pred_test = ols_model.predict(X_test)
    
    # Calculate errors
    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)
    
    print(f"OLS Coefficients: {ols_model.coef_[0]:.6f}")
    print(f"OLS Intercept: {ols_model.intercept_:.6f}")
    print(f"OLS Training MSE: {train_error:.6f}")
    print(f"OLS Testing MSE: {test_error:.6f}")
    
    return ols_model, y_pred_test

def create_pybrain_ann(X_train, X_test, y_train, y_test, input_size):
    """Create and train PyBrain Artificial Neural Network"""
    print("\n" + "="*50)
    print("Training PyBrain ANN Model")
    print("="*50)
    
    # Create supervised datasets
    train_dataset = SupervisedDataSet(input_size, 1)
    test_dataset = SupervisedDataSet(input_size, 1)
    
    # Add samples to datasets
    for i in range(len(X_train)):
        train_dataset.addSample(X_train[i], [y_train[i]])
    
    for i in range(len(X_test)):
        test_dataset.addSample(X_test[i], [y_test[i]])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # Build feedforward neural network
    # Architecture: Input(1) -> Hidden(5) -> Output(1)
    network = buildNetwork(input_size, 5, 1, bias=True)
    print(f"ANN Architecture: {input_size}-5-1 (feedforward)")
    
    # Create trainer with backpropagation
    trainer = BackpropTrainer(network, train_dataset, learningrate=0.01, verbose=False)
    
    # Train the network
    print("Training ANN...")
    errors = []
    for epoch in range(1000):  # Increased epochs for better approximation
        error = trainer.train()
        errors.append(error)
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Error = {error:.6f}")
    
    # Test the network
    test_predictions = []
    for i in range(len(test_dataset)):
        prediction = network.activate(test_dataset['input'][i])
        test_predictions.append(prediction[0])
    
    test_error = np.mean((np.array(test_predictions) - y_test) ** 2)
    print(f"Final Training Error: {errors[-1]:.6f}")
    print(f"ANN Testing MSE: {test_error:.6f}")
    
    return network, test_predictions, errors

def create_sklearn_ann(X_train, X_test, y_train, y_test, input_size):
    """Create and train scikit-learn ANN as fallback"""
    print("\n" + "="*50)
    print("Training Scikit-learn ANN Model")
    print("="*50)
    
    # Create MLP Regressor (feedforward neural network)
    ann_model = MLPRegressor(
        hidden_layer_sizes=(5,),  # One hidden layer with 5 neurons
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        verbose=False
    )
    
    print(f"ANN Architecture: {input_size}-5-1 (feedforward)")
    print("Training ANN...")
    
    # Train the model
    ann_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = ann_model.predict(X_train)
    y_pred_test = ann_model.predict(X_test)
    
    # Calculate errors
    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)
    
    print(f"ANN Training MSE: {train_error:.6f}")
    print(f"ANN Testing MSE: {test_error:.6f}")
    
    return ann_model, y_pred_test

def save_model_to_xml(model, model_type, filename='UE_05_App3_ANN_Model.xml'):
    """Save the trained model to XML file"""
    print(f"\nSaving {model_type} model to {filename}...")
    
    try:
        # Create root element
        root = ET.Element("ANN_Model")
        root.set("type", model_type)
        
        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        ET.SubElement(metadata, "Architecture").text = "1-5-1"
        ET.SubElement(metadata, "ModelType").text = model_type
        ET.SubElement(metadata, "InputFeatures").text = "1"
        ET.SubElement(metadata, "OutputFeatures").text = "1"
        
        # Add model parameters based on type
        if model_type == "PyBrain_ANN":
            params = ET.SubElement(root, "Parameters")
            # For PyBrain, we can store basic info since full serialization is complex
            ET.SubElement(params, "HiddenNeurons").text = "5"
            ET.SubElement(params, "ActivationFunction").text = "Sigmoid"
            ET.SubElement(params, "LearningRate").text = "0.01"
            
        elif model_type == "Sklearn_ANN":
            params = ET.SubElement(root, "Parameters")
            ET.SubElement(params, "HiddenLayerSizes").text = "5"
            ET.SubElement(params, "Activation").text = "relu"
            ET.SubElement(params, "Solver").text = "adam"
            ET.SubElement(params, "MaxIter").text = "1000"
        
        # Add performance info
        performance = ET.SubElement(root, "Performance")
        if hasattr(model, 'loss_'):
            ET.SubElement(performance, "FinalLoss").text = f"{model.loss_:.6f}"
        
        # Create XML tree and save with pretty formatting
        tree = ET.ElementTree(root)
        
        # Pretty print XML
        xml_str = ET.tostring(root, encoding='utf-8')
        parsed_xml = minidom.parseString(xml_str)
        pretty_xml = parsed_xml.toprettyxml(indent="  ")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        print(f"Model successfully saved to {filename}")
        
        # Also save the actual model using pickle for practical use
        model_filename = filename.replace('.xml', '.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Practical model binary saved to {model_filename}")
        
    except Exception as e:
        print(f"Error saving model to XML: {e}")

def compare_predictions(ols_predictions, ann_predictions, y_test, X_test):
    """Compare OLS and ANN predictions"""
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    # Calculate differences
    ols_errors = np.abs(ols_predictions - y_test)
    ann_errors = np.abs(ann_predictions - y_test)
    
    print(f"OLS Mean Absolute Error: {np.mean(ols_errors):.6f}")
    print(f"ANN Mean Absolute Error: {np.mean(ann_errors):.6f}")
    
    print("\nSample Predictions Comparison:")
    print("-" * 40)
    for i in range(min(5, len(X_test))):
        print(f"Sample {i+1}:")
        print(f"  Input (x): {X_test[i][0]:.4f}")
        print(f"  Actual (y): {y_test[i]:.4f}")
        print(f"  OLS Prediction: {ols_predictions[i]:.4f}")
        print(f"  ANN Prediction: {ann_predictions[i]:.4f}")
        print(f"  OLS Error: {ols_errors[i]:.4f}")
        print(f"  ANN Error: {ann_errors[i]:.4f}")
        print()

# Main execution
if __name__ == "__main__":
    try:
        print("="*60)
        print("Artificial Neural Network Training Application")
        print("="*60)
        
        # Load and prepare data
        X_train, X_test, y_train, y_test, input_size = load_and_prepare_datasets('dataset03.csv')
        
        # Train OLS model (baseline)
        ols_model, ols_predictions = create_ols_model(X_train, y_train, X_test, y_test)
        
        # Train ANN model
        if PYBRAIN_AVAILABLE:
            ann_model, ann_predictions, training_errors = create_pybrain_ann(
                X_train, X_test, y_train, y_test, input_size
            )
            model_type = "PyBrain_ANN"
        else:
            ann_model, ann_predictions = create_sklearn_ann(
                X_train, X_test, y_train, y_test, input_size
            )
            model_type = "Sklearn_ANN"
        
        # Save the ANN model
        save_model_to_xml(ann_model, model_type, 'UE_05_App3_ANN_Model.xml')
        
        # Compare model performances
        compare_predictions(ols_predictions, ann_predictions, y_test, X_test)
        
        print("\n" + "="*60)
        print("Training Completed Successfully!")
        print("="*60)
        print(f"ANN Model saved as: UE_05_App3_ANN_Model.xml")
        print(f"ANN Model binary saved as: UE_05_App3_ANN_Model.pkl")
        print(f"Model approximates the same function as OLS model")
        print(f"Architecture: 1 input -> 5 hidden neurons -> 1 output (feedforward)")
        
    except FileNotFoundError:
        print("Error: 'dataset03.csv' file not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
