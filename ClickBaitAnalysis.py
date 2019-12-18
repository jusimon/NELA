import warnings
import pickle

def prediction(xtest):
    pickleModel = "/content/gdrive/My Drive/AlternusVeraDataSets2019/FinalExam/Drifters/julian/Click_Bait_Model.pkl"
    pickle_in = open(pickleModel, "rb")
    loadData = pickle.load(pickle_in)
    predicedProb = loadData.predict_proba([xtest])[:,1]
    return predicedProb[0]
            
class ClickBait:
    def __init__(self, xtest):
        self.x_test = xtest
    def predict(self):
        return prediction(self.x_test)
