import numpy as np

def sigmoid(x):
    """
    sigmoid 함수

    Arguments:
        x:  scalar 또는 numpy array

    Return:
        s:  sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))

    return s

def relu(x):
    """
    ReLU 함수

    Arguments:
        x : scalar 또는 numpy array

    Return:
        s : relu(x)
    """
    s = np.maximum(0,x)
    
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
        self.nlayer = len(layerDims)-1

        self.parameters = self.weightInit(layerDims)
        self.grads = {}
        self.vel = {}
        self.s = {}
        self.cache = {}
        self.initialize_optimizer()

    def weightInit(self, layerDims):

        np.random.seed(1)
        parameters = {}

        for l in range(1, len(layerDims)):
            parameters['W' + str(l)] = np.random.randn(int(layerDims[l]),int(layerDims[l-1]))*0.01
            parameters['b' + str(l)] = np.zeros((int(layerDims[l]),1))

        return parameters

    # iniitialize parameter for optimizer
    def initialize_optimizer(self):

        for l in range(1,self.nlayer+1):
            self.vel['sdW'+str(l)]=np.zeros(self.parameters['W'+str(l)].shape)
            self.vel['sdb'+str(l)]=np.zeros(self.parameters['b'+str(l)].shape)


    def forward(self, X):
        '''
        forward propagation

        Arguments:
            X: input data

        Return:
            A23: network output
        '''

        ## 코딩시작 

        W1, b1, W2, b2, W3, b3 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'],self.parameters['b2'], self.parameters['W3'],self.parameters['b3']
      
        Z1 = np.dot(W1,X)+b1
        A1 = relu(Z1)
        Z2 = np.dot(W2,A1)+b2
        A2 = relu(Z2)
        Z3 = np.dot(W3,A2)+b3
        A3 = sigmoid(Z3)

        self.cache.update(X=X, Z1=Z1, A1=A1, Z2=Z2, A2=A2, Z3=Z3, A3=A3)

        return A3

    def backward(self,lambd=0.7):
        '''
        regularization term이 추가된 backward propagation.

        Arguments:
            lambd

        Return:
        '''
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

        self.grads.update(dW1=dW1, db1= db1, dW2=dW2, db2=db2, dW3=dW3, db3=db3)


        return


    def compute_cost(self, A3, Y, lambd=0.7):
 
        self.cache.update(Y=Y)
        W1, W2, W3 = self.parameters["W1"], self.parameters["W2"], self.parameters["W3"]

        logprobs = -(Y*np.log(A3)+(1-Y)*np.log(1-A3))
        cost = (1/self.nSample)*np.sum(logprobs) + (lambd/2/self.nSample)*np.sum(W1*W1) + (lambd/2/self.nSample)*np.sum(W2*W2) + (lambd/2/self.nSample)*np.sum(W3*W3)

        cost = float(np.squeeze(cost))

        assert(isinstance(cost, float))
        
        return cost

    def update_params(self, learning_rate=1.2, beta2=0.999, epsilon=1e-8):
        '''
        backpropagation을 통해 얻은 gradients를 update한다.

        Arguments:
            learning_rate:  학습할 learning rate

        Return:
        '''
        W1,b1,W2,b2,W3,b3 =self.parameters['W1'],self.parameters['b1'],self.parameters['W2'],self.parameters['b2'],self.parameters['W3'],self.parameters['b3']
        dW1, db1, dW2, db2, dW3, db3 =self.grads['dW1'],self.grads['db1'],self.grads['dW2'],self.grads['db2'],self.grads['dW3'],self.grads['db3']
        sdW1, sdb1, sdW2, sdb2, sdW3, sdb3=self.vel['sdW1'],self.vel['sdb1'],self.vel['sdW2'],self.vel['sdb2'],self.vel['sdW3'],self.vel['sdb3']

        sdW1=beta2*sdW1+(1-beta2)*dW1*dW1
        sdb1=beta2*sdb1+(1-beta2)*db1*db1
        sdW2=beta2*sdW2+(1-beta2)*dW2*dW2
        sdb2=beta2*sdb2+(1-beta2)*db2*db2
        sdW3=beta2*sdW3+(1-beta2)*dW3*dW3
        sdb3=beta2*sdb3+(1-beta2)*db3*db3


        W1 = W1-learning_rate*dW1/np.sqrt(sdW1+epsilon)
        b1 = b1-learning_rate*db1/np.sqrt(sdb1+epsilon)
        W2 = W2-learning_rate*dW2/np.sqrt(sdW2+epsilon)
        b2 = b2-learning_rate*db2/np.sqrt(sdb2+epsilon)
        W3 = W3-learning_rate*dW3/np.sqrt(sdW3+epsilon)
        b3 = b3-learning_rate*db3/np.sqrt(sdb3+epsilon)

        self.parameters.update(W1=W1, b1= b1, W2=W2, b2=b2, W3=W3, b3=b3)

        return 

    def predict(self,X):
        '''
        학습한 network가 잘 학습했는지, test set을 통해 확인한다.

        Arguments:
            X: input data
        Return:
        '''

        A3 = self.forward(X)
        predictions=np.where(A3<0.5,0,1) 

        return predictions