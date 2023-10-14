from LLMToolkit.data_processing import DataLoader

def test_data_loader():
    loader = DataLoader()
    data_path = "tests/data_processing/sample_datasets/sample_data.txt"
    data = loader.load(data_path)
    assert len(data) == 4
    print("DataLoader Test Passed!")

test_data_loader()
