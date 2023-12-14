# make_dataset.py
import pathlib
import yaml
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

# main() function is a special function in python that is used as entry point for a programm.When a Python script is run, the interpreter looks for the main() function and executes it. If the main() function is not defined, the interpreter will start executing the program from the top.
def main():

#The pathlib.Path(__file__) expression in Python returns a Path object representing the absolute path to the file that is currently being executed.
    curr_dir = pathlib.Path(__file__)

#The pathlib.Path().parent attribute returns the parent directory of the current file.
    home_dir = curr_dir.parent.parent.parent

#The as_posix() method is used to return a string representation of the path with forward slashes (/). This is useful for making paths portable across different operating systems, as Windows uses backslashes (\) as path separators, while most other operating systems use forward slashes.
    params_file = home_dir.as_posix() + '/params.yaml'

#The yaml.safe_load function is used to read YAML files with the PyYAML library. It parses the YAML document from the file and returns the corresponding Python data structure.
    params = yaml.safe_load(open(params_file))["make_dataset"]

    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'

    data = load_data(data_path)
    train_data, test_data = split_data(data,params['test_split', params['seed']])
    save_data(train_data,test_data,output_path)

if __name__ == "__main__":

    # main() function is only executed when the script is run directly, and not when it is imported as a module.
    main()class