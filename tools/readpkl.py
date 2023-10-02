import pickle
f = open('results.pkl','rb')
data = pickle.load(f)
print(data)