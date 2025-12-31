import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse
import torch
import logging
from os.path import join
from datetime import datetime
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Parameters & Cuda environment
# =============================================================================
kwargs = {'num_workers': 0, 'pin_memory': True}
no_cuda=False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--penaltycoef", type=int, default=1e-5)
parser.add_argument("--upperlr", type=int, default=1e-2)
parser.add_argument("--lowerlr", type=int, default=1e-2)
parser.add_argument("--printfrequency",   type=str, default=100)
args = parser.parse_args()


# logging setup

now_time = datetime.now() 
time = now_time.strftime('%Y-%m-%d-%H-%M-%S')
current_dir = os.getcwd() 
logging_file = join(current_dir,"logs/", "time="+str(time)+".log")
logging.basicConfig(filename=logging_file, level=logging.INFO)

# =============================================================================
# Evaluate the performance on the test set
# =============================================================================
def Evaluation(testX, testY, model, criterion, epoch):
    testX = torch.from_numpy(testX).float()
    testX = testX.to(device)
    with torch.no_grad():
        y_pred=model(testX.float())
    error_test = mean_squared_error(y_pred.cpu(),testY)
    return error_test

def Evaluation_classification(testX, testY, model, epoch):
    testX = torch.from_numpy(testX).float()
    testX = testX.to(device)
    correct = 0
    total = len(testY)
    testY = torch.tensor(np.reshape(testY, len(testY)))
    with torch.no_grad():
        y_pred= (model(testX.float())).data.cpu()
    _, predicted = y_pred.max(1)
    correct += predicted.eq(testY).sum().item()
    acc = correct / total
    return acc


def Meta_Additive_models(train_loader, validation_loader, testX, testY,total_dimension,task):
    if task=='regression':
        # create lower level model
        model = build_model(total_dimension,task='regression')
        # build upper level model   A single-level MLP with 100 hidden nodes
        weighting_net = UpperModel(1, 10, 1).cuda()
        # define loss function (criterion) and optimizer
        criterion = nn.MSELoss().cuda()
        Epoches=20000
        for epoch in range(Epoches):
            regression_train(train_loader, validation_loader, model, weighting_net, epoch)
            # evaluate on testing set
            test_error = Evaluation(testX, testY, model, criterion, epoch)
            if epoch %args.printfrequency==0:
                print(" Test Error: ",test_error)  
        logging.info("the test MSE is :{}".format(test_error))
        logging.info("the best weights is :{}".format(model.predict.weight))
        return weighting_net
    else:
        # create lower level model
        model = build_model(total_dimension,task='classification',num_classes=2)
        # build upper level model   A single-level MLP with 100 hidden nodes
        weighting_net = UpperModel(1, 10, 1).cuda()
        # define loss function (criterion) and optimizer
        Epoches=10000
        for epoch in range(Epoches):
            classification_train(train_loader, validation_loader, model, weighting_net, epoch)
            # evaluate on testing set
            acc = Evaluation_classification(testX, testY, model, epoch)
            if epoch %args.printfrequency==0:
                print(" Test Acc: {acc:.3f}".format(acc=acc))
        logging.info("the test accuracy is :{}".format(acc))
        logging.info("the best weights is :{}".format(model.predict.weight))
        return weighting_net
            

# =============================================================================
# The sparsity-induced regularization  
# Two strategies:   
#      Directly introduce the penalty into the loss ; 
#      Proximally update the parameter     
#           See:   Consistent Feature Selection for Analytic Deep Neural Networks, NIPS 2020, Dinh, Vu, Ho, Lam Si Tung
# =============================================================================
def penalty(w,total_dimension,splines_d):
    # Group Lasso style penalty on variable group
    # W: R^ Pd*1 -> R^ P * d      add L2,1 penalty on each variable group R^d
    return torch.sum(torch.norm(torch.reshape(w,(total_dimension//splines_d,splines_d)),p=2,dim=0))

def proximal(w, lam, eta):
    # Lasso style penalty on each individual variable
    tmp = torch.norm(w, dim=1) - lam*eta
    alpha = torch.clamp(tmp, min=0)
    v = torch.nn.functional.normalize(w, dim=1)*alpha[:,None]
    w.data = v

def proximal_g(w, lam, eta,total_dimension,splines_d):#d=3 for regression
    print(w.shape)
    # Group Lasso style penalty on variable group
    # W: R^ Pd*1 -> R^ P * d      add L2,1 penalty on each variable group R^d
    a=torch.reshape(torch.norm(torch.reshape(w,(total_dimension//splines_d,splines_d)),p=2,dim=1),((total_dimension//splines_d),1))
    b = torch.cat((a,a,a), dim=1)
    c = torch.reshape(b,(1,total_dimension))
    tmp = 1 - lam*eta/c
    alpha = torch.clamp(tmp, min=0)
    v = w*alpha
    # print(v.shape)
    w.data = v
    
def proximal_gc(w, lam, eta,total_dimension,splines_d): #d=5 for classification
    # print(w.shape)
    # Group Lasso style penalty on variable group
    # W: R^ Pd*1 -> R^ P * d      add L2,1 penalty on each variable group R^d
    a=torch.reshape(torch.norm(torch.reshape(w,(total_dimension//splines_d,splines_d)),p=2,dim=1),(total_dimension//splines_d,1))
    b = torch.cat((a,a,a,a,a), dim=1)
    c = torch.reshape(b,(2,total_dimension//2))
    tmp = 1 - lam*eta/c
    alpha = torch.clamp(tmp, min=0)
    v = w*alpha
    w.data = v
    
# =============================================================================
# The key steps for updating the parameters of MAM for regression
# =============================================================================
def step1(input,target,model,vnet,epoch):
    input_var, target_var = input.to(device), target.to(device)
    input_var = input_var.to(torch.float32)
    target_var = target_var.to(torch.float32)
    
    # The inner level model : 1-layer linear 
    meta_model = build_model(input.size(1),task='regression')
    meta_model.load_state_dict(model.state_dict())
    
    
    y_f_hat = meta_model(input_var)
    
    cost = F.mse_loss(y_f_hat,target_var, reduce=False) 
    cost_v = torch.reshape(cost, (len(cost), 1))
    v_lambda = vnet(cost_v.data)
    
    # l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v) + args.penaltycoef*penalty(meta_model.predict.weight,300) 
    l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v) 
    meta_model.zero_grad()
    grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
    # args.lowerlr = args.lowerlr * ((0.1 ** int(epoch >= 2000)) * (0.1 ** int(epoch >= 4000)))
    # args.upperlr = args.upperlr * ((0.1 ** int(epoch >= 2000)) * (0.1 ** int(epoch >= 4000)))
    meta_model.update_params(lr_inner=args.lowerlr, source_params=grads)
    # proximal(meta_model.predict.weight, lam=args.penaltycoef, eta=args.lowerlr)
# =============================================================================
#     The proximal operation indeed is to calculate the gradient of the penalty
# =============================================================================
    proximal_g(meta_model.predict.weight, lam=args.penaltycoef, eta=args.lowerlr,total_dimension=300,splines_d=3)
    # del grads    
    return meta_model,input_var,target_var,l_f_meta

def step2(validation_loader,meta_model,optimizer_vnet,epoch):
    input_validation, target_validation = next(iter(validation_loader))
    input_validation, target_validation=input_validation.to(device), target_validation.to(device)
    input_validation, target_validation=input_validation.to(torch.float32), target_validation.to(torch.float32)

    y_g_hat = meta_model(input_validation)
    target_validation_var=target_validation.cuda()
    
    
    l_g_meta = F.mse_loss(y_g_hat, target_validation_var)
    # l_g_meta.backward(retain_graph=True)
    prec_meta = mean_squared_error(y_g_hat.data.cpu(), target_validation_var.data.cpu())


    optimizer_vnet.zero_grad()
    l_g_meta.backward()
    optimizer_vnet.step()
    return prec_meta

def step3(input_var,target_var,model,vnet,optimizer_a,epoch):
    y_f = model(input_var)
    cost_w = F.mse_loss(y_f,target_var, reduce=False)
    cost_v = torch.reshape(cost_w, (len(cost_w), 1))
    prec_train = mean_squared_error (y_f.data.cpu(), target_var.data.cpu())
    
    with torch.no_grad():
        w_new = vnet(cost_v)
    norm_v = torch.sum(w_new)
    
    if norm_v != 0:
        w_v = w_new / norm_v
    else:
        w_v = w_new
    
    # l_f = torch.sum(cost_v * w_v)/len(cost_v) + args.penaltycoef*penalty(model.predict.weight,300) 
    l_f = torch.sum(cost_v * w_v)/len(cost_v)
    optimizer_a.zero_grad()
    l_f.backward()
    optimizer_a.step()
    # proximal(model.predict.weight, lam=args.penaltycoef, eta=args.lowerlr,total_dimension=300,splines_d=3)
    proximal_g(model.predict.weight, lam=args.penaltycoef, eta=args.lowerlr,total_dimension=300,splines_d=3)
    return prec_train
    
def regression_train(train_loader, validation_loader,model, vnet,epoch):
    """Train for one epoch on the training set"""
    
    optimizer_a = torch.optim.SGD(model.params(), args.lowerlr)
    optimizer_vnet = torch.optim.SGD(vnet.params(), args.upperlr)
    # Load new training data for updating the lower level parameters
    for i, (input, target) in enumerate(train_loader): 
        
        model.train()
        # Step1: Update \hat{\beta}(\theta)
        meta_model,input_var,target_var,loss_weighted = step1(input,target,model,vnet,epoch)
        # Step2: Update \theta
        prec_meta  = step2(validation_loader,meta_model,optimizer_vnet,epoch)
        # Step3: Update \beta
        prec_train=step3(input_var,target_var,model,vnet,optimizer_a,epoch)
    if(epoch%args.printfrequency==0):    
        print('Epoch: [{0}]\t''Train Loss {loss:.4f} \t  Weighted Train Loss {wloss:.4f} \t''Meta Loss {meta_loss:.4f} \t'.format(epoch,loss=prec_train,wloss=loss_weighted,meta_loss=prec_meta))
    return vnet


# =============================================================================
# The key steps for updating the parameters of MAM for classification
# =============================================================================
# class MAM_logistic_mean(nn.Module):  # Single output version--> Set  LowerModel_classification output=1
#     def __init__(self):
#         super().__init__()
#     def forward(self, x, y):
#         return torch.mean(torch.log(1+torch.exp(x))- x*y)

# class MAM_logistic(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x, y):
#         return torch.log(1+torch.exp(x))- x*y
    
def step11(input,target,model,vnet,epoch):
    # print(target)
    input_var, target_var = input.to(device='cuda:0'), target.to(device='cuda:0')
    input_var = input_var.to(torch.float32)
    target_var = target_var.to(torch.int64)
    target_var = torch.reshape(target_var,(1,len(target_var)))[0]

    # The inner level model : 1-layer linear 
    meta_model = build_model(input.size(1),task="classification",num_classes=2)
    meta_model.load_state_dict(model.state_dict())
    
    
    y_f_hat = meta_model(input_var).cuda()
    y_f_hat = y_f_hat.to(torch.float)
    cost = F.cross_entropy(y_f_hat,target_var, reduce=False)
    cost_v = torch.reshape(cost, (len(cost), 1))
    v_lambda = vnet(cost_v.data)
    # l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v) + args.penaltycoef*penalty(meta_model.predict.weight,1000)
    l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v) 
    # print( F.cross_entropy(y_f_hat,target_var) )
    meta_model.zero_grad()
    grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
    # Decreasing step sizes  suggested for Imbalanced and multi-objective cases
    # args.lowerlr = args.lowerlr * ((0.1 ** int(epoch >= 2000)) * (0.1 ** int(epoch >= 4000)))
    # args.upperlr = args.upperlr * ((0.1 ** int(epoch >= 2000)) * (0.1 ** int(epoch >= 4000)))
    meta_model.update_params(lr_inner=args.lowerlr, source_params=grads)  #Hypergradient
    # proximal(meta_model.predict.weight, lam=args.penaltycoef, eta=args.lowerlr)
    proximal_gc(meta_model.predict.weight, lam=args.penaltycoef, eta=args.lowerlr,total_dimension=1000,splines_d=5)
    # del grads    
    return meta_model,input_var,target_var,l_f_meta

def step22(validation_loader,meta_model,optimizer_vnet,epoch):
    input_validation, target_validation = next(iter(validation_loader))
    input_validation, target_validation=input_validation.to(device), target_validation.to(device)
    input_validation, target_validation=input_validation.to(torch.float32), target_validation.to(torch.int64)
    y_g_hat = meta_model(input_validation)
    y_g_hat = y_g_hat.to(torch.float)
    target_validation_var=target_validation.cuda()
    target_validation_var = torch.reshape(target_validation_var,(1,len(target_validation_var)))[0]
    l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
    optimizer_vnet.zero_grad()
    l_g_meta.backward()
    optimizer_vnet.step()
    return l_g_meta

def step33(input_var,target_var,model,vnet,optimizer_a,epoch):
    y_f = model(input_var)
    cost_w = F.cross_entropy(y_f,target_var, reduce=False)
    cost_v = torch.reshape(cost_w, (len(cost_w), 1))
    prec_train = F.cross_entropy(y_f,target_var)
    with torch.no_grad():
        w_new = vnet(cost_v)
    norm_v = torch.sum(w_new)
    if norm_v != 0:
        w_v = w_new / norm_v
    else:
        w_v = w_new
    # l_f = torch.sum(cost_v * w_v)/len(cost_v) + args.penaltycoef*penalty(model.predict.weight,1000)
    l_f = torch.sum(cost_v * w_v)/len(cost_v) 
    model.zero_grad()
    l_f.backward()
    optimizer_a.step()
    # proximal(model.predict.weight, lam=args.penaltycoef, eta=args.lowerlr)
    proximal_gc(model.predict.weight, lam=args.penaltycoef, eta=args.lowerlr,total_dimension=1000,splines_d=5)
    return prec_train

def classification_train(train_loader, validation_loader,model, vnet,epoch):
    """Train for one epoch on the training set"""
    
    optimizer_a = torch.optim.SGD(model.params(), args.lowerlr)
    optimizer_vnet = torch.optim.SGD(vnet.params(), args.upperlr)
    
    # Load new training data for updating the lower level parameters
    for i, (input, target) in enumerate(train_loader): 
        
        model.train()
        # Step1: Update \hat{\beta}(\theta)
        meta_model,input_var,target_var,wloss = step11(input,target,model,vnet,epoch)
        # Step2: Update \theta
        prec_meta  = step22(validation_loader,meta_model,optimizer_vnet,epoch)
        # Step3: Update \beta
        prec_train =step33(input_var,target_var,model,vnet,optimizer_a,epoch)
    if(epoch%args.printfrequency==0):    
        print('Epoch: [{0}]\t''Train Loss {loss:.4f} \t Weighted Train Loss {loss2:.4f} \t''Val Loss {meta_loss:.4f} \t'.format(epoch,loss=prec_train,loss2=wloss,meta_loss=prec_meta))
    return vnet


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


# =============================================================================
# The update process for updating the meta-relevant parameters 
# adopted from: Adrien Ecoffet https://github.com/AdrienLE
# =============================================================================
class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


# =============================================================================
# The upper model indeed is a single layer with 100 hidden activation nodes and
# Sigmoid output activation
# =============================================================================
class UpperModel(MetaModule):
    def __init__(self, input, hidden1, output):
        super(UpperModel, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)

def build_model(dimension,task,num_classes=2):
    if (task == 'regression'):
        model = LowerModel(dimension,1)
    else:
        model = LowerModel_classification(dimension,num_classes)
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    return model

# =============================================================================
# Notice that the lower model for regression indeed is a linear operator  and the loss function
# is defined before with sparsity-induced (group lasso) penalty   
# =============================================================================
class LowerModel(MetaModule):
    def __init__(self,n_feature,n_output):
        super(LowerModel,self).__init__()
        self.predict = MetaLinear(n_feature,n_output)
    def forward(self,x):
        out = self.predict(x)
        return out
    
# =============================================================================
# The model for classification : single linear layer + sigmoid:
#       f(x) = \beta x + b
#       Then we have: P(y=1|x) = exp(f(x)) / (1+exp(f(x))) = 1/(1+exp(-f(x))) = sigmoid (f(x))
# =============================================================================
class LowerModel_classification(MetaModule):
    def __init__(self,n_feature,n_output):
        super( LowerModel_classification,self).__init__()
        self.predict = MetaLinear(n_feature,n_output)
    def forward(self,x):
        out = self.predict(x)
        return out
