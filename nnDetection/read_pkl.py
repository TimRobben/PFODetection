import pickle

objects = []
with (open("/home/tarobben/scratch/nndet/Task001_test/preprocessed/D3V001_3d.pkl", "rb")) as openfile:
    while True:
        try:
            print(pickle.load(openfile))
        except EOFError:
            break

