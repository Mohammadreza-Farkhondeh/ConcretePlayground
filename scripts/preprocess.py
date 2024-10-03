import pandas as pd
import argparse
import numpy as np

def parse_args():
    """
    Set up the argument parser to handle command-line input for the script.
    This allows users to specify the path to their input CSV file and 
    the directory where the processed output files will be saved.
    
    Returns:
        args: Namespace containing the input file path and output directory path.
    """
    parser = argparse.ArgumentParser(description="Preprocess concrete compressive strength data.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file containing concrete data.')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the processed data as CSV files.')
    return parser.parse_args()


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    This function checks for missing values and applies strategies to handle them.
    Common strategies include filling missing values with the mean or median,
    or dropping rows or columns with excessive missing data.
    
    Args:
        df: DataFrame containing the raw data.
    
    Returns:
        df: DataFrame with missing values handled.
    """
    # Check for missing values
    print("Missing values before handling:")
    print(df.isnull().sum())

    # Here we drop columns with more than 30% missing values and fill the rest with the median
    threshold = 0.3 * len(df)
    df = df.dropna(thresh=threshold, axis=1)  # Drop columns that don't meet the threshold
    df.fillna(df.median(), inplace=True)  # Fill remaining missing values with the median
    
    print("Missing values after handling:")
    print(df.isnull().sum())
    
    return df

def main():
    # Parse the command line arguments to get user-defined paths for input and output
    args = parse_args()
    
    # Load the dataset from the specified CSV file using pandas.
    df = pd.read_csv(args.input)
    
    # Handle any missing values in the dataset.
    df = handle_missing_values(df)
    
    # Prepare the feature set (X) by dropping the target variable column from the DataFrame.
    X = df.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
    
    # The target variable (y) is extracted as a separate Series.
    y = df['Concrete compressive strength(MPa, megapascals) ']
    
    X["Age (day)"] = np.log(X["Age (day)"])

    # Save the processed features and targets to CSV files for future use.
    X_df = pd.DataFrame(X, columns=X.columns)
    y_df = pd.DataFrame(y)
    
    # Save the processed data to the specified output directory.
    X_df.to_csv(f"{args.output}/X_scaled.csv", index=False)
    y_df.to_csv(f"{args.output}/y.csv", index=False)

    print("Data preprocessing complete and saved to output directory.")

if __name__ == "__main__":
    # Execute the main function when the script is run directly.
    main()

