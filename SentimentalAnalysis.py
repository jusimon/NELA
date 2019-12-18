import warnings
import pickle

def prediction(xtest):
    pickleModel = "/content/gdrive/My Drive/AlternusVeraDataSets2019/FinalExam/Drifters/julian/SentimentAnalysis_Model.pkl"
    pickle_in = open(pickleModel, "rb")
    loadData = pickle.load(pickle_in)
    predicedProb = loadData.predict_proba([text])[:,1]
    return predicedProb
            
class Sentimental:
    def __init__(self, xtest):
        self.x_test = xtest
    def predict(self):
        return prediction(self.x_test)
