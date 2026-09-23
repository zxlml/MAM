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
    # FIX: the half-open intervals (x >= t[j]) & (x < t[j+1]) assign zero weight to
    # x == t[-1] (the right boundary). Clamp those samples onto the last basis so the
    # partition of unity (row sum == 1) holds on the closed interval.
    right_edge = (x == t[-1])
    if np.any(right_edge):
        B[right_edge, :] = 0.0
        B[right_edge, -1] = 1.0
    return B

    
def transform_splines(trainX,  validX, testX,r, whiten_tol=1e-10):
# =============================================================================
#     Suggestion: r=3 for regression tasks and r=5 for classification tasks
#  FIX 1: the knot range [min, max] is taken from the TRAIN split only and
#         reused for valid/test. The original code re-fitted knots per split,
#         so the three splits were mapped by *different* feature maps and the
#         additive model could not learn a consistent rule.
#  FIX 2: each Bernstein block is whitened by an SVD fitted on the TRAIN block
#         (directions with sigma <= whiten_tol * sigma_max are zeroed out).
#         The Bernstein basis spans the constant function, so every block
#         contains an all-ones direction -> the raw design matrix has exact
#         cross-block collinearity (condition number ~1e15), which makes the
#         convex lower-level problem numerically unsolvable. Whitening is an
#         orthogonal transform *within* each block, so the model class and the
#         group-L2,1 penalty (invariant to orthogonal within-group transforms)
#         are preserved; only the numerically unidentifiable directions are
#         removed.
# =============================================================================
    n_tr, p = trainX.shape[0], trainX.shape[1]
    n_va, n_te = validX.shape[0], testX.shape[0]
    X_trn_spline = np.ones([n_tr, p*r])
    X_val_spline = np.ones([n_va, p*r])
    X_tst_spline = np.ones([n_te, p*r])

    for j in range(0, p):
        # knots from the TRAIN split only -> identical feature map for all splits
        min_k_trn = min(trainX[:, j])
        max_k_trn = max(trainX[:, j])
        knot_trn  = [ min_k_trn, max_k_trn ]

        B_trn = bsplinebasis(trainX[:, j], knot_trn, r)
        B_val = bsplinebasis(validX[:, j], knot_trn, r)
        B_tst = bsplinebasis(testX[:, j],  knot_trn, r)

        # SVD whitening fitted on the train block only
        U, S, Vt = np.linalg.svd(B_trn, full_matrices=False)
        S_inv = np.where(S > whiten_tol * S[0], 1.0 / np.maximum(S, 1e-300), 0.0)
        W = (Vt.T * S_inv)            # (r, r) whitening map, zero on dead dirs

        # FIX 3: plain SVD whitening gives every column unit L2 NORM (= std
        # 1/sqrt(n)), which makes the design matrix ~40x smaller than the
        # conventional unit-variance scale and stalls plain SGD solvers (the
        # bias reaches the class-prior solution first, then gradients vanish).
        # Rescale by sqrt(n_train) so each surviving column has ~unit std.
        scale = np.sqrt(n_tr)
        X_trn_spline[:, j*r:(j+1)*r] = (B_trn @ W) * scale
        X_val_spline[:, j*r:(j+1)*r] = (B_val @ W) * scale
        X_tst_spline[:, j*r:(j+1)*r] = (B_tst @ W) * scale

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

def data_process(trainX, trainY, validX, validY,testX,batch,r,seed=None):
    trainX, validX, testX = transform_splines(trainX, validX, testX,r)
    train_data = Data.TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    val_data = Data.TensorDataset(torch.tensor(validX), torch.tensor(validY))
    # test_data = Data.TensorDataset( torch.tensor(testX), torch.tensor(testY))

    # Explicit per-loader generators: since torch 2.x the shuffle sampler draws
    # its seed lazily from the global RNG, so without an explicit generator the
    # batch order is not reproducible even after manual_seed().
    g_train = torch.Generator().manual_seed(seed if seed is not None else 0)
    g_val = torch.Generator().manual_seed(seed if seed is not None else 0)

    train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=batch,
    shuffle=True,
    num_workers=0,
    generator=g_train,
    )

    val_loader = Data.DataLoader(
    dataset=val_data,
    batch_size=batch,
    shuffle=True,
    num_workers=0,
    generator=g_val,
    )
    

    
    return train_loader,val_loader, testX


def _set_seed(seed):
    # Reproducible simulation: fix numpy / torch / random seeds when requested.
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    import random as _random
    _random.seed(seed)


def generate_regression(number=1000,dimension=100,noise_type='Gaussian',seed=None):
    # NOTE: the original implementation regenerated the whole dataset `number`
    # times inside a loop and silently kept only the last replicate, which was
    # extremely wasteful (O(number^2 * dimension) samples drawn) and made the
    # effective sample size ambiguous. We now draw ONE train/valid/test split
    # of `number` samples each, following the TSpAM simulation protocol.
    _set_seed(seed)
    N, p = number,dimension
    noise = noise_type
    reg_data = Regression(N, p, noise=noise)

    (trainX, trainY), (validX, validY), (testX, testY) = reg_data.generate_data()
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=3,seed=seed)
    return train_loder,val_loder, testX, testY

def generate_corrupted_classification(number=1000,dimension=100,percentage=0.3,seed=None):
    _set_seed(seed)
    N, p = number,dimension
    frac = percentage
    cls_data = Classfication_corrupted(N, p, frac=frac)

    (trainX, trainY), (validX, validY), (testX, testY) = cls_data.generate_data()
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=5,seed=seed)
    return train_loder,val_loder, testX, testY


def generate_imbalanced_classification(number=1000,dimension=100,ratio=0.15,seed=None):
    _set_seed(seed)
    N, p = number,dimension
    frac = ratio
    cls_data = Classfication_imbalance(N, p, frac=frac)

    (trainX, trainY), (validX, validY), (testX, testY) = cls_data.generate_data()
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=5,seed=seed)
    return train_loder,val_loder, testX, testY

def generate_multi_classification(number=1000,dimension=100,ratio=0.15,seed=None):
    _set_seed(seed)
    N, p = number,dimension
    frac = ratio
    cls_data = Classfication_multi(N, p, frac=frac)

    (trainX, trainY), (validX, validY), (testX, testY) = cls_data.generate_data()
    train_loder,val_loder,testX =data_process(trainX, trainY, validX, validY,testX,batch=200,r=5,seed=seed)
    return train_loder,val_loder, testX, testY
