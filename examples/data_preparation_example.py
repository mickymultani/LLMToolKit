from llmtoolkit.data_processing import load_data_from_txt, split_data

def main():
    # Load data from a text file
    data = load_data_from_txt("sample_data.txt")

    # Split the data into training, validation, and test sets
    train_data, val_data, test_data = split_data(data)

    # Print some statistics
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

if __name__ == "__main__":
    main()
