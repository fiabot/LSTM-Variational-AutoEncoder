from ptb import PTB 
if __name__ == "__main__":
    data_dir = "VGDLData"

    training = PTB(data_dir, "train",create_data=True)
    test = PTB(data_dir, "test",create_data=True)
    valid = PTB(data_dir, "valid",create_data=True)