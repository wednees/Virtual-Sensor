import pickle
from main_page import main_component

with open('../ML/model.pkl', 'rb') as file:
    model = pickle.load(file)
    
main_component(model)


