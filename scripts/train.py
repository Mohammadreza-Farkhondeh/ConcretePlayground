import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
import argparse

def parse_args():
    """
    Set up the argument parser to handle command-line input for the script.
    This allows users to specify the directory where the preprocessed data is located
    and where to save the trained model.

    Returns:
        args: Namespace containing the input directory path and the model save path.
    """
    parser = argparse.ArgumentParser(description="Train a neural network model for concrete compressive strength prediction.")
    parser.add_argument('--input-dir', type=str, required=True, help='Directory path where preprocessed data is stored.')
    parser.add_argument('--save-dir', type=str, default='model_js', help='Directory path to save the trained TensorFlow.js model.')
    return parser.parse_args()

def load_data(input_dir):
    """
    Load the preprocessed dataset from CSV files.

    This function retrieves the feature set and the corresponding target values 
    and ensures they are in the correct format (NumPy arrays) for training the model.

    Args:
        input_dir: Directory path where the preprocessed CSV files are stored.

    Returns:
        X: Feature array and corresponding target array.
    """
    column_order = [
        'Cement (component 1)(kg in a m^3 mixture)',
        'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
        'Fly Ash (component 3)(kg in a m^3 mixture)',
        'Water  (component 4)(kg in a m^3 mixture)',
        'Superplasticizer (component 5)(kg in a m^3 mixture)',
        'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
        'Fine Aggregate (component 7)(kg in a m^3 mixture)',
        'Age (day)'
    ]
    df = pd.read_csv(f"{input_dir}/concrete_compressive_strength.csv")

    X = df[column_order].values
    y = df["Concrete compressive strength(MPa, megapascals) "].values

    return X, y

def build_model(input_shape):
    """
    Construct a Keras Sequential model designed for regression tasks.

    This model uses dense layers to learn complex relationships in the data.
    The architecture consists of input, hidden, and output layers. The choice of 
    activation functions and the number of units can significantly affect the model's performance.

    Args:
        input_shape: The number of input features, which determines the shape of the input layer.

    Returns:
        nn_model: The compiled Keras model ready for training.
    """
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),  # Input layer defined by the number of features
        tf.keras.layers.Dense(64, activation='relu'),  # First hidden layer with 64 units and ReLU activation for non-linearity
        tf.keras.layers.Dense(64, activation='relu'),  # Second hidden layer with the same configuration for deeper learning
        tf.keras.layers.Dense(1)  # Output layer for regression, predicting a single continuous value
    ])
    
    # Compile the model using the Adam optimizer and mean squared error loss function.
    nn_model.compile(optimizer='adam', loss='mean_squared_error')

    return nn_model

def main():
    # Parse the command line arguments to get the input and save directory
    args = parse_args()
    input_dir = args.input_dir  # Get the input directory from the command line arguments
    save_dir = args.save_dir     # Get the save directory from the command line arguments

    # Load the preprocessed data, ensuring it's in a format suitable for training.
    X, y = load_data(input_dir)

    # Build the neural network model by passing the number of features in the dataset.
    model = build_model(X.shape[1])  # Shape[1] gives the number of input features

    # Train the model on the entire dataset since we are not using any test split.
    model.fit(X, y, epochs=50, batch_size=32)

    # Save the trained model in a specified directory for TensorFlow.js.
    tfjs.converters.save_keras_model(model, save_dir)  # Use the save directory from the command-line arguments

    print("Model training and conversion complete.")

if __name__ == "__main__":
    # Run the main function when the script is executed.
    main()

