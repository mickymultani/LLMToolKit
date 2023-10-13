from llmtoolkit.data_processing import load_data_from_txt

def test_load_data_from_txt():
    # Let's create a dummy .txt file
    with open("dummy.txt", 'w') as file:
        file.write("Hello\nWorld")

    data = load_data_from_txt("dummy.txt")
    assert data == ["Hello\n", "World"], f"Expected ['Hello\\n', 'World'], but got {data}"
