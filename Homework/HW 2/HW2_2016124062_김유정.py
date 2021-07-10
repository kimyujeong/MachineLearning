import numpy as np
import matplotlib.pyplot as plt
from data_utils import decision_boundary, generate_dataset
import sklearn
import sklearn.datasets
np.random.seed(1)


def sigmoid(x):
    """
    sigmoid 함수

    Arguments:
        x:  scalar 또는 numpy array

    Return:
        s:  sigmoid(x)
    """

    ## 코딩시작
    s = 1/(1+np.exp(-x))
    ## 코딩 끝

    return s

def relu(x):
    """
    ReLU 함수

    Arguments:
        x : scalar 또는 numpy array

    Return:
        s : relu(x)
    """
    ## 코딩시작
    s = np.where(x>0,x,0) ##np.maximum(0,x)
    ## 코딩 끝

    return s

class NeuralNetwork:
    def __init__(self,layerDims, nSample):
        '''
        학습할 네트워크.

        Arguments:
            layerDims [array]: layerDims[i] 는 레이어 i의 hidden Unit의 개수 (layer0 = input layer)
            nSample: 데이터셋의 샘플 수
        '''

        self.nSample = nSample
        self.parameters = self.weightInit(layerDims)
        self.grads = {}
        self.cache = {}
        
    def weightInit(self, layerDims):
        """
        network parameter 초기화

        Arguments:
            layerDims [array] : 
        
        Returns:
            parameters: network의 parameter를 dictionary 형태로 저장
                        key 값은  "W1", "b1", ..., "WL", "bL"
                        
        Tips: 0.01..... 꼭....
        
        """
     
        np.random.seed(1)
        parameters = {}
        
        ## 코딩 시작
        # parameter를 초기화 하세요.
        for l in range(1, len(layerDims)):
            parameters['W' + str(l)] = np.random.randn(int(layerDims[l]),int(layerDims[l-1]))*0.01
            parameters['b' + str(l)] = np.zeros((int(layerDims[l]),1))
        ## 코딩 끝

        return parameters

    def forward(self, X):
        '''
        forward propagation

        Arguments:
            X: input data

        Return:
            A23: network output
        '''

        ## 코딩시작 
        # parameter 값을 불러와서, Z1, A1, Z2, A2, Z3, A3 계산후 cache update
        W1, b1, W2, b2, W3, b3 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'],self.parameters['b2'], self.parameters['W3'],self.parameters['b3']
      
        Z1 = np.dot(W1,X)+b1
        A1 = relu(Z1)
        Z2 = np.dot(W2,A1)+b2
        A2 = relu(Z2)
        Z3 = np.dot(W3,A2)+b3
        A3 = sigmoid(Z3)

        self.cache.update(X=X, Z1=Z1, A1= A1, Z2=Z2, A2=A2, Z3=Z3, A3=A3)

        ## 코딩 끝

        return A3

    def backward(self):
        '''
        backward propagation. gradients를 구한다.

        Arguments:

        Return:
        '''
        ## 코딩 시작
        # parameter와 cache를 이용해 gradients를 구한후 grads update한다.
        # dZ1, dZ2, dZ3, dW1, dW2, dW3, db1, db2, db3, dA1, dA2 를 각각 구하세요.

        W1, b1, W2, b2, W3, b3 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'],self.parameters['b2'], self.parameters['W3'],self.parameters['b3']
        X, Y, dZ1, A1, dZ2, A2, dZ3, A3 = self.cache['X'], self.cache['Y'],self.cache['Z1'],self.cache['A1'],self.cache['Z2'],self.cache['A2'],self.cache['Z3'],self.cache['A3']
        Z1, Z2 = self.cache['Z1'],self.cache['Z2']

        dZ3 = A3-Y
        dW3 = (1/self.nSample)*np.dot(dZ3,A2.T)
        db3 = (1/self.nSample)*np.sum(dZ3,axis=1,keepdims=True)
        dZ2 = np.dot(W3.T,dZ3)*np.where(Z2<0,0,1)
        dW2 = (1/self.nSample)*np.dot(dZ2,A1.T)
        db2 = (1/self.nSample)*np.sum(dZ2,axis=1,keepdims=True)
        dZ1 = np.dot(W2.T,dZ2)*np.where(Z1<0,0,1)
        dW1 = (1/self.nSample)*np.dot(dZ1,X.T)
        db1 = (1/self.nSample)*np.sum(dZ1,axis=1,keepdims=True)

        self.grads.update(dW1=dW1, db1= db1, dW2=dW2, db2=db2, dW3=dW3, db3=db3)

        ## 코딩 끝



        return

    def update_params(self, learning_rate=1.2):
        '''
        backpropagation을 통해 얻은 gradients를 update한다.

        Arguments:
            learning_rate:  학습할 learning rate

        Return:
        '''

        ## 코딩시작
        # parameter와 grads를 이용해 gradients descents를 통해 새로운 weight를 구하고, parameters update

        W1,b1,W2,b2,W3,b3 =self.parameters['W1'],self.parameters['b1'],self.parameters['W2'],self.parameters['b2'],self.parameters['W3'],self.parameters['b3']
        dW1, db1, dW2, db2, dW3, db3 =self.grads['dW1'],self.grads['db1'],self.grads['dW2'],self.grads['db2'],self.grads['dW3'],self.grads['db3']

        W1 = W1-learning_rate*dW1
        b1 = b1-learning_rate*db1
        W2 = W2-learning_rate*dW2
        b2 = b2-learning_rate*db2
        W3 = W3-learning_rate*dW3
        b3 = b3-learning_rate*db3

        self.parameters.update(W1=W1, b1= b1, W2=W2, b2=b2, W3=W3, b3=b3)

        ## 코딩 끝
        
        return 

    def compute_cost(self, A3, Y):
        '''
        cross-entropy loss를 이용하여 cost를 구한다.

        Arguments:
            A2 : network 결과값
            Y  : 정답 label(groud truth)
        Return:
            cost
        '''

        self.cache.update(Y=Y)
        
        ## 코딩시작
        logprobs = -(Y*np.log(A3)+(1-Y)*np.log(1-A3))
        cost = (1/self.nSample)*np.sum(logprobs)

        ### 코딩끝
        
        cost = float(np.squeeze(cost))  

        assert(isinstance(cost, float))
        
        return cost


    def compute_cost_with_regularization(self, A3, Y, lambd=0.7):
        '''
        cross-entropy loss에 regularization term을 이용하여 cost를 구한다.

        Arguments:
            A3 : network 결과값
            Y  : 정답 label(groud truth)
            lambd : 람다 값. 
        Return:
            cost
        '''

        self.cache.update(Y=Y)
        W1, W2, W3 = self.parameters["W1"], self.parameters["W2"], self.parameters["W3"]

        ## 코딩시작

        logprobs = -(Y*np.log(A3)+(1-Y)*np.log(1-A3))
        cost = (1/self.nSample)*np.sum(logprobs) + (lambd/2/self.nSample)*np.sum(W1*W1) + (lambd/2/self.nSample)*np.sum(W2*W2) + (lambd/2/self.nSample)*np.sum(W3*W3)

        ### 코딩끝
        
        cost = float(np.squeeze(cost))  

        assert(isinstance(cost, float))
        
        return cost

    def backward_with_regularization(self,lambd=0.7):
        '''
        regularization term이 추가된 backward propagation.

        Arguments:
            lambd: 

        Return:
        '''
        ## 코딩 시작
        # regularization term이 추가된 cost에서 back-propagation을 진행, grads update

        W1, b1, W2, b2, W3, b3 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'],self.parameters['b2'], self.parameters['W3'],self.parameters['b3']
        X, Y, dZ1, A1, dZ2, A2, dZ3, A3 = self.cache['X'], self.cache['Y'],self.cache['Z1'],self.cache['A1'],self.cache['Z2'],self.cache['A2'],self.cache['Z3'],self.cache['A3']
        Z1,Z2=self.cache['Z1'],self.cache['Z2']
        dZ3 = A3-Y
        dW3 = (1/self.nSample)*np.dot(dZ3,A2.T) +(lambd/self.nSample)*W3
        db3 = (1/self.nSample)*np.sum(dZ3,axis=1,keepdims=True)
        dZ2 = np.dot(W3.T,dZ3)*np.where(Z2<0,0,1)
        dW2 = (1/self.nSample)*np.dot(dZ2,A1.T) +(lambd/self.nSample)*W2
        db2 = (1/self.nSample)*np.sum(dZ2,axis=1,keepdims=True)
        dZ1 = np.dot(W2.T,dZ2)*np.where(Z1<0,0,1)
        dW1 = (1/self.nSample)*np.dot(dZ1,X.T) +(lambd/self.nSample)*W1
        db1 = (1/self.nSample)*np.sum(dZ1,axis=1,keepdims=True)


        ## dZ3 = A3 - Y
        ## dW3 = 1./self.nSample * np.dot(dZ3,A2.T)+lambd/self.nSample*W3
        ## db3 = 1./self.nSample * np.sum(dZ3,axis=1,keepdims=True)

        ## dA2 = np.dot(W3.T,dZ3)
        ## dZ2 = np.multiply(dA2,np.int64(A2>0))
        ## dW2 = 1./self.nSample * np.dot(dZ2,A1.T)+lambd/self.nSample*W2
        ## db2 = 1./self.nSample * np.sum(dZ2,axis=1,keepdims=True)

        ## dA1
        ## dZ1
        ## dW1
        ## db1

        self.grads.update(dW1=dW1, db1= db1, dW2=dW2, db2=db2, dW3=dW3, db3=db3)
        
        ## 코딩


        return

    def predict(self,X):
        '''
        학습한 network가 잘 학습했는지, test set을 통해 확인한다.

        Arguments:
            X: input data
        Return:
        '''
        ## 코딩시작
        # network 출력이 0.5보다 크면 1 아니면 0, 결과를 predictions로 출력한다.

        A3 = self.forward(X)
        predictions=np.where(A3<0.5,0,1) 

        ## 코딩 끝

        return predictions

def main():
    np.random.seed(1)
    num_iterations=12000
    learning_rate =0.3

    ### dataset loading 하기.
    X_train,Y_train, X_test, Y_test = generate_dataset()
    # plt.title("Data distribution")
    # plt.scatter(X_train[0, :], X_train[1, :], c=Y_train[0,:], s=20, cmap=plt.cm.RdBu)
    # plt.show()

    
    ## 코딩 시작
    nSample = 200 ## X_train.shape[1]
    layerDims = np.array([[2],[20],[7],[1]]) ## [2,20,7,1]
    ## 코딩 끝

    simpleNN = NeuralNetwork(layerDims,nSample)
    training(X_train, Y_train, simpleNN, num_iterations, learning_rate)

    simpleNN_R = NeuralNetwork(layerDims,nSample)
    training_with_regularization(X_train, Y_train, simpleNN_R, num_iterations, learning_rate)

    ### print prediction
    predictions = simpleNN.predict(X_train)
    print ('Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')
    decision_boundary(lambda x: simpleNN.predict(x.T), X_train, Y_train, name="Train Dataset")
    predictions = simpleNN.predict(X_test)
    print ('Accuracy: %d' % float((np.dot(Y_test,predictions.T) + np.dot(1-Y_test,1-predictions.T))/float(Y_test.size)*100) + '%')
    decision_boundary(lambda x: simpleNN.predict(x.T), X_test, Y_test, name="test Dataset")


    predictions = simpleNN_R.predict(X_train)
    print ('Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')
    decision_boundary(lambda x: simpleNN_R.predict(x.T), X_train, Y_train, name="Train Dataset")
    predictions = simpleNN_R.predict(X_test)
    print ('Accuracy: %d' % float((np.dot(Y_test,predictions.T) + np.dot(1-Y_test,1-predictions.T))/float(Y_test.size)*100) + '%')
    decision_boundary(lambda x: simpleNN_R.predict(x.T), X_test, Y_test, name="test Dataset")

def training(X_train, Y_train, simpleNN, num_iterations, learning_rate):

    ## 코딩시작
    # num_iterations 동안 training regularization이 없는 simple neural network를 학습하는 code를 작성하세요.
    # return 값으론, 학습된 모델을 출력하세요.
    
    for i in range(0, num_iterations):
        A3 = simpleNN.forward(X_train)               
        cost = simpleNN.compute_cost(A3,Y_train)     
        simpleNN.backward()                          
        simpleNN.update_params()  ##simpleNN.update_params(learning_rate=learning_rate)                  
        if i % 1000 ==0:
            print("(No Regularization) Cost after iteration %i: %f" %(i,cost))

    ##코딩 끝 

    # predictions = simpleNN.predict(X_train)
    # print ('Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')
    # decision_boundary(lambda x: simpleNN.predict(x.T), X_train, Y_train, name="Train Dataset")

    return simpleNN
 
def training_with_regularization(X_train, Y_train, simpleNN, num_iterations, learning_rate):

    ## 코딩시작
    # num_iterations 동안 training regularization이 있는 simple neural network를 학습하는 code를 작성하세요.
    # return 값으론, 학습된 모델을 출력하세요.
    
    for i in range(0, num_iterations):
        A3 = simpleNN.forward(X_train)              
        cost = simpleNN.compute_cost_with_regularization(A3,Y_train)     
        simpleNN.backward_with_regularization()       
        simpleNN.update_params()  ##simpleNN.update_params(learning_rate=learning_rate)                 
        if i % 1000 ==0:
            print("(Regularization) Cost after iteration %i: %f" %(i,cost))

    ##코딩 끝 
    
    # predictions = simpleNN.predict(X_train)
    # print ('Accuracy: %d' % float((np.dot(Y_train,predictions.T) + np.dot(1-Y_train,1-predictions.T))/float(Y_train.size)*100) + '%')
    # decision_boundary(lambda x: simpleNN.predict(x.T), X_train, Y_train, name="Train Dataset")

    return simpleNN
if __name__=='__main__':
    main()