import numpy as np
import matplotlib.pyplot as plt
import dataUtils
from model_2020125001 import NeuralNetwork

import json
from json import JSONEncoder
import numpy

np.random.seed(1)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# save weights of model from json file
def saveParams(params, path):
    with open(path, "w") as make_file:
        json.dump(params, make_file, cls=NumpyArrayEncoder)
    print("Done writing serialized NumPy array into file")

# load weights of model from json file
def loadParams(path):
    with open(path, "r") as read_file:
        print("Converting JSON encoded data into Numpy array")
        decodedArray = json.load(read_file)
    return decodedArray


def main():
    epochs = 100
    learning_rate = 1e-5
    batch_size = 32 
    resume = False # path of model weights
    model_weights_path = './weights_2016124062.json'

    ### dataset loading 하기.
    dataPath = 'dataset/train'
    valPath = 'dataset/val'
    dataloader = dataUtils.Dataloader(dataPath, minibatch=batch_size)
    val_dataloader = dataUtils.Dataloader(valPath)
    
    nSample = batch_size
    layerDims = [7500, 64, 32, 1]

    simpleNN = NeuralNetwork(layerDims, nSample)
    if resume:
        simpleNN.parameters = loadParams(resume)

    for epoch in range(1, epochs):
        training(dataloader, simpleNN, learning_rate, epoch)
        
        if epoch%10==1:
            validation(val_dataloader,simpleNN)
            
    validation(val_dataloader,simpleNN)
    saveParams(simpleNN.parameters, model_weights_path)


def validation(dataloader, simpleNN):
    for i, (images, targets) in enumerate(dataloader):
        print(images.shape)
        print(targets.shape)
        # do validation
        A3 = simpleNN.forward(images)
        cost = simpleNN.compute_cost(A3,targets)
        # simpleNN.backward()
        # simpleNN.update_params()
        
        if i % 10 ==0:
            print("(Validation) Cost after iteration %i: %f" %(i,cost))


def training(dataloader, simpleNN, learning_rate, epoch):

    for i, (images, targets) in enumerate(dataloader):
        # do training

        A3 = simpleNN.forward(images)               
        cost = simpleNN.compute_cost(A3,targets)     
        simpleNN.backward()                          
        simpleNN.update_params()
        print(A3.shape)
        if i % 1000 ==0:
            print("(Training) Cost after iteration %i: %f" %(i,cost))


    return simpleNN
 

if __name__=='__main__':
    main()