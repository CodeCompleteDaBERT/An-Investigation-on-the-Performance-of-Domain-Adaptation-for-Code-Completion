import pickle


methods = []
with open("methods.pickle", "rb") as f:
    methods = pickle.load(f)

dataset_size = len(methods)
testing_size = dataset_size // 10
evalation_size = dataset_size // 10
training_size = dataset_size - testing_size - evalation_size

training_methods = methods[:training_size]
evalation_methods = methods[training_size : training_size + evalation_size]
testing_methods = methods[training_size + evalation_size :]
with open("training_methods.pickle", mode="wb") as f:
    pickle.dump(training_methods, f)
with open("evalation_methods.pickle", mode="wb") as f:
    pickle.dump(evalation_methods, f)
with open("testing_methods.pickle", mode="wb") as f:
    pickle.dump(testing_methods, f)
