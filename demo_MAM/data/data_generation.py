# =============================================================================
# #The data generation for regression and classification tasks are mainly based on the settings of TSpAM:
# # Tilted Sparse Additive Models ICML2023 Yingjie Wang et.al.
# =============================================================================


import numpy as np
import torch
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from random import sample
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
import matplotlib as plt

def bsplineBasis_j(x, t, j, M):
    #  bsplineBasis: compute a bspline basis function of order m.
    p = M-1; # polynomial degree
    
    # construct the b-spline recursively.
    if(p == 0): # piecewise constatnt
        Bj = ((x >= t[j]) & (x < t[j+1]))
    else:
        # If the two knots in the denominator are equal, we ignore the term (to not devide by zero).
        denom1 = t[j + M - 1] - t[j];
        if(denom1 == 0):
            Bj = np.zeros(len(x))
        else:
            Bj = (x - t[j]) * bsplineBasis_j(x, t, j, M-1) / denom1;
        
        denom2 = t[j + M] - t[j + 1]
        if(denom2 != 0):
            Bj = Bj + (t[j + M] - x) * bsplineBasis_j(x, t, j+1, M-1) / denom2;
    return Bj

def bsplinebasis(x, t, M):   
    m = len(x)
    # repeat the first and the last knots m times
    t1=list(t[0]*np.ones(M-1))
    t2=list(t[-1]*np.ones(M-1))
    t=t1+t+t2
    # j th basis function : 1 <= j <= length(t) - M -1 ; There are K + 2*M knots ; M = length(t) - 1;
    B = np.zeros([m, len(t) - M])
    for j in range(0,len(t)- M): #is the same as j=1 : K+M, K is the number of interior knots
        B[:,j] = bsplineBasis_j( x, t, j, M )# j-1 0 <= j <= len(tt) - n - 2.
    # print(B)
    return B

    
def transform_splines(trainX,  validX, testX,r):
# =============================================================================
#     Suggestion: r=3 for regression tasks and r=5 for classification tasks
# =============================================================================
    X_trn_spline = np.ones([trainX.shape[0],trainX.shape[1]*r])
    X_val_spline = np.ones([validX.shape[0],validX.shape[1]*r])
    X_tst_spline = np.ones([testX.shape[0],testX.shape[1]*r])
    
    for j in range(0,len(trainX[0])):
      min_k_trn = min(trainX[:,j])
      max_k_trn = max(trainX[:,j])
      knot_trn  = [ min_k_trn,max_k_trn]
      X_trn_spline[:,(j)*r:(j+1)*r] = bsplinebasis(trainX[:,j], knot_trn, r)
      # print(X_trn_spline[:,j*r:(j+1)*r])
                    
      min_k_val = min(validX[:,j])
      max_k_val = max(validX[:,j])
      knot_val  = [ min_k_val,max_k_val]
      X_val_spline[:,(j)*r:(j+1)*r] = bsplinebasis(validX[:,j], knot_val, r)
      # print(X_val_spline[:,j*r:(j+1)*r])
            
      min_k_tst = min(testX[:,j])
      max_k_tst = max(testX[:,j])
      knot_tst  = [ min_k_tst,max_k_tst]
      X_tst_spline[:,(j)*r:(j+1)*r] = bsplinebasis(testX[:,j], knot_tst, r)
      # print(X_tst_spline[:,j*r:(j+1)*r])
    
    return X_trn_spline,X_val_spline,X_tst_spline

class Regression:
    def __init__(self, N, p, noise="None"):
        self.N, self.p = N, p
        self.noise = noise
    
    def add_noise(self, dataY):
        N = len(dataY)
        noise = self.noise
        if noise == "studentT":
            noise = np.random.standard_t(df=2, size=(N, 1))
        elif noise == "mixGauss":
            noise = np.zeros((N, 1))
            for i in range(N):
                c = np.random.uniform()
                if c < 0.8:
                    noise[i][0] = np.random.normal(loc=-2, scale=1)
                else:
                    noise[i][0] = np.random.normal(loc=40, scale=1)
        elif noise == "mean":
            noise = np.zeros((N, 1))
            for i in range(N):
                c = np.random.uniform()
                if c < 0.8:
                    noise[i][0] = np.random.normal(loc=-2, scale=1)
                else:
                    noise[i][0] = np.random.normal(loc=8, scale=1)
        elif noise == "modal":
            noise = np.zeros((N, 1))
            for i in range(N):
                c = np.random.uniform()
                if c < 0.8:
                    noise[i][0] = np.random.normal(loc=0, scale=1)
                else:
                    noise[i][0] = np.random.normal(loc=20, scale=1)
        elif noise == "Gaussian":
            noise = np.random.randn(N, 1)
        elif noise == "Gauss2":
            noise = np.random.normal(loc=0, scale=2, size=(N, 1))
        elif noise == "chiSquare":
            noise = np.random.chisquare(1, size=(N, 1))
        elif noise == "None":
            noise = 0
        noiseY = dataY + noise
        # print("max y is :", np.max(np.abs(dataY)), "max noise is :", np.max(np.abs(noise)))
        return noiseY
    
    def plot_data(self, dataX, noiseY):
        dataY = self.generate_Y(dataX)
        plt.plot(range(self.N), dataY)
        plt.plot(range(self.N), noiseY)
        plt.savefig( "trainY.png")
        plt.show()
    
    def generate_Y(self, dataX):
        f1 = -2 * np.sin(2 * dataX[:, 0])
        f2 = 8 * np.square(dataX[:, 1])
        f3 = 7 * np.sin(dataX[:, 2]) / (2-np.sin(dataX[:,2]))
        f4 = 6 * np.exp(-dataX[:, 3])
        f5 = np.power(dataX[:, 4], 3) + 1.5*np.square(dataX[:, 4] - 1)
        f6 = 5 * dataX[:, 5]
        f7 = 10 * np.sin(np.exp(-dataX[:, 6]/2))
        f8 = -10 * norm.cdf(dataX[:, 7], loc=0.5, scale=0.8)
    
        Y = f1+f2+f3+f4+f5+f6+f7+f8
        Y = np.expand_dims(Y, axis=1)
        return Y


    def generate_data(self):
        trainX = np.random.uniform(-1, 1, size=(self.N, self.p))
        validX = np.random.uniform(-1, 1, size=(self.N, self.p))
        testX = np.random.uniform(-1, 1, size=(self.N, self.p))

        trainY = self.generate_Y(trainX)
        validY = self.generate_Y(validX)
        testY  = self.generate_Y(testX)

        # add noise to trainY 
        trainY = self.add_noise(trainY)

        scaler1 = StandardScaler()
        scaler1.fit(np.vstack((trainX, validX)))
        trainX, validX, testX = map(scaler1.transform, [trainX, validX, testX])
        scaler2 = StandardScaler()
        scaler2.fit(np.vstack((trainY, validY)))
        trainY, validY, testY = map(scaler2.transform, [trainY, validY, testY])
        
        return (trainX, trainY), (validX, validY), (testX, testY)
        
class Classfication_corrupted:
    def __init__(self, N, p, frac=0.1):
        self.N, self.p = N, p
        self.frac = frac
    
    def add_noise(self, dataY):
        N = len(dataY)
        frac = self.frac
        idx = sample(range(self.N), int(self.N*self.frac))
        dataY[idx] = 1 - dataY[idx]
        return dataY

    def plot_data(self, dataX, noiseY):
        dataY = self.generate_Y(dataX)
        pos_idx = np.where(dataY==1)[0]
        neg_idx = np.where(dataY==-1)[0]
        plt.scatter(dataX[pos_idx, 0], dataX[pos_idx, 1], c='green')
        plt.scatter(dataX[neg_idx, 0], dataX[neg_idx, 1], c='red')
        plt.savefig("trainY.png")
        plt.show()
    
    def generate_Y(self, dataX):
        f1 = np.square(dataX[:, 0] - 0.5)
        f2 = np.square(dataX[:, 1] - 0.5)
        Y = f1 + f2 - 0.08
        Y = np.expand_dims(Y, axis=1)
        Y[Y>0] = 1
        Y[Y<=0] = 0
        return Y.astype(int)

    def generate_X(self):
        N, p = self.N, self.p
        W = np.random.uniform(low=0, high=1, size=(N, p))
        U = np.random.uniform(low=0, high=1, size=(N, 1))
        return (W+U) / 2

    def generate_data(self):
        trainX = self.generate_X()
        validX = self.generate_X()
        testX = self.generate_X()

        trainY = self.generate_Y(trainX)
        validY = self.generate_Y(validX)
        testY  = self.generate_Y(testX)

        # add noise to trainY 
        trainY = self.add_noise(trainY)

        # plot data
        # self.plot_data(trainX, trainY)

        # standard data
        scaler1 = StandardScaler()
        scaler1.fit(np.vstack((trainX, validX)))
        trainX, validX, testX = map(scaler1.transform, [trainX, validX, testX])
        
        return (trainX, trainY), (validX, validY), (testX, testY)

class Classfication_imbalance:
    def __init__(self, N, p, frac=0.1):
        self.N, self.p = N, p
        self.frac = frac
    
    def sample_data(self, dataX, dataY, frac):
        neg_num = int(self.N * frac)
        pos_num = self.N - neg_num
        idx1 = sample(list(np.where(dataY==0)[0]), neg_num)
        idx2 = sample(list(np.where(dataY==1)[0]), pos_num)
        negX, negY = dataX[idx1], dataY[idx1]
        posX, posY = dataX[idx2], dataY[idx2]
        trainX = np.vstack((negX, posX))
        trainY = np.vstack((negY, posY))
        data = np.hstack((trainX, trainY))
        np.random.shuffle(data)
        trainX, trainY = data[:,:-1], data[:, -1]
        return trainX, np.expand_dims(trainY, axis=1)

    def plot_data(self, dataX, noiseY):
        dataY = self.generate_Y(dataX)
        pos_idx = np.where(dataY==1)[0]
        neg_idx = np.where(dataY==-1)[0]
        plt.scatter(dataX[pos_idx, 0], dataX[pos_idx, 1], c='green')
        plt.scatter(dataX[neg_idx, 0], dataX[neg_idx, 1], c='red')
        plt.savefig( "trainY.png")
        plt.show()
    
    def generate_Y(self, dataX):
        f1 = np.square(dataX[:, 0] - 0.5)
        f2 = np.square(dataX[:, 1] - 0.5)
        Y = f1 + f2 - 0.08
        Y = np.expand_dims(Y, axis=1)
        Y[Y>0] = 1
        Y[Y<=0] = 0
        return Y.astype(int)

    def generate_X(self):
        N, p = 10000, self.p
        W = np.random.uniform(low=0, high=1, size=(N, p))
        U = np.random.uniform(low=0, high=1, size=(N, 1))
        return (W+U) / 2

    def generate_data(self):
        trainX = self.generate_X()
        validX = self.generate_X()
        testX = self.generate_X()

        trainY = self.generate_Y(trainX)
        validY = self.generate_Y(validX)
        testY  = self.generate_Y(testX)

        trainX, trainY = self.sample_data(trainX, trainY, self.frac)
        validX, validY = self.sample_data(validX, validY, 0.5)
        testX,  testY  = self.sample_data(testX, testY, 0.5)
        # print(trainY.shape)

        # standard data
        scaler1 = StandardScaler()
        scaler1.fit(np.vstack((trainX, validX)))
        trainX, validX, testX = map(scaler1.transform, [trainX, validX, testX])

        return (trainX, trainY), (validX, validY), (testX, testY)


class Classfication_multi:
    def __init__(self, N, p, frac=0.1):
        self.N, self.p = N, p
        self.frac = frac

    def plot_data(self, dataX, noiseY):
        dataY = self.generate_Y(dataX)
        pos_idx = np.where(dataY==1)[0]
        neg_idx = np.where(dataY==-1)[0]
        plt.scatter(dataX[pos_idx, 0], dataX[pos_idx, 1], c='green')
        plt.scatter(dataX[neg_idx, 0], dataX[neg_idx, 1], c='red')
        plt.savefig("trainY.png")
        plt.show()
    
    def generate_X(self):
        N, p = 10000, self.p
        W = np.random.uniform(low=0, high=1, size=(N, p))
        U = np.random.uniform(low=0, high=1, size=(N, 1))
        return (W+U) / 2
    
    def generate_Y(self, dataX):
        f1 = np.square(dataX[:, 0] - 0.5)
        f2 = np.square(dataX[:, 1] - 0.5)
        Y = f1 + f2 - 0.08
        Y = np.expand_dims(Y, axis=1)
        Y[Y>0] = 1
        Y[Y<=0] = 0
        return Y.astype(int)
    
    def add_noise(self, dataY, frac2):
        N = len(dataY)
        if frac2>0:
            idx = sample(range(N), int(N*frac2))
            dataY[idx] = 1 - dataY[idx]
        return dataY.astype(int)
    
    def sample_data(self, dataX, dataY, frac, frac2=0):

        ## sample data accodring to fraction
        neg_num = int(self.N * frac)
        pos_num = self.N - neg_num
        idx1 = sample(list(np.where(dataY==0)[0]), neg_num)
        idx2 = sample(list(np.where(dataY==1)[0]), pos_num)
        negX, negY = dataX[idx1], dataY[idx1]
        posX, posY = dataX[idx2], dataY[idx2]

        ## add noise to Y
        negY = self.add_noise(negY, frac2)
        posY = self.add_noise(posY, frac2)

        trainX = np.vstack((negX, posX))
        trainY = np.vstack((negY, posY))
        data = np.hstack((trainX, trainY))
        np.random.shuffle(data)
        trainX, trainY = data[:,:-1], data[:, -1]
        return trainX, np.expand_dims(trainY, axis=1)

    def generate_data(self):
        trainX = self.generate_X()
        validX = self.generate_X()
        testX = self.generate_X()

        trainY = self.generate_Y(trainX)
        validY = self.generate_Y(validX)
        testY  = self.generate_Y(testX)

        trainX, trainY = self.sample_data(trainX, trainY, self.frac, frac2=0.3) #Imbalance  & Corruption = 0.3
        validX, validY = self.sample_data(validX, validY, 0.5, frac2=0)
        testX,  testY  = self.sample_data(testX, testY, 0.5, frac2=0)
        # print(trainY.shape)

        # standard data
        scaler1 = StandardScaler()
        scaler1.fit(np.vstack((trainX, validX)))
        trainX, validX, testX = map(scaler1.transform, [trainX, validX, testX])

        return (trainX, trainY), (validX, validY), (testX, testY)

def data_process(trainX, trainY, validX, validY,testX,batch,r):
    trainX, validX, testX = transform_splines(trainX, validX, testX,r)
    train_data = Data.TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    val_data = Data.TensorDataset(torch.tensor(validX), torch.tensor(validY))
    # test_data = Data.TensorDataset( torch.tensor(testX), torch.tensor(testY))
    
    train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=batch,
    shuffle=True,
    num_workers=0,
    )

    val_loader = Data.DataLoader(
    dataset=val_data,
    batch_size=batch,
    shuffle=True,
    num_workers=0,
    )
    

    
    return train_loader,val_loader, testX


def generate_regression(number=1000,dimension=100,noise_type='Gaussian'):
    N, p = number,dimension
    noise = noise_type
    reg_data = Regression(N, p, noise=noise)

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x,  test_y  = [], []

    for i in range(number):
        (trainX, trainY), (validX, validY), (testX, testY) = reg_data.generate_data()
        train_x.append(trainX)
        train_y.append(trainY)
        valid_x.append(validX)
        valid_y.append(validY)
        test_x.append(testX)
        test_y.append(testY)
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=3)
    return train_loder,val_loder, testX, testY

def generate_corrupted_classification(number=1000,dimension=100,percentage=0.3):
    N, p = number,dimension
    frac = percentage
    reg_data = Classfication_corrupted(N, p, frac=frac)

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x,  test_y  = [], []


    for i in range(number):
        (trainX, trainY), (validX, validY),  (testX, testY) = reg_data.generate_data()
        train_x.append(trainX)
        train_y.append(trainY)
        valid_x.append(validX)
        valid_y.append(validY)
        test_x.append(testX)
        test_y.append(testY)
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=5)
    return train_loder,val_loder, testX, testY


def generate_imbalanced_classification(number=1000,dimension=100,ratio=0.15):
    N, p = number,dimension
    frac = ratio
    reg_data = Classfication_imbalance(N, p, frac=frac)

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x,  test_y  = [], []


    for i in range(number):
        (trainX, trainY), (validX, validY),  (testX, testY) = reg_data.generate_data()
        train_x.append(trainX)
        train_y.append(trainY)
        valid_x.append(validX)
        valid_y.append(validY)
        test_x.append(testX)
        test_y.append(testY)
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=5)
    return train_loder,val_loder,testX, testY

def generate_multi_classification(number=1000,dimension=100,ratio=0.15):
    N, p = number,dimension
    frac = ratio
    reg_data = Classfication_multi(N, p, frac=frac)

    train_x, train_y = [], []
    valid_x, valid_y = [], []
    test_x,  test_y  = [], []

    for i in range(number):
        (trainX, trainY), (validX, validY), (testX, testY) = reg_data.generate_data()
        train_x.append(trainX)
        train_y.append(trainY)
        valid_x.append(validX)
        valid_y.append(validY)
        test_x.append(testX)
        test_y.append(testY)
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=5)
    return train_loder,val_loder, testX, testY
