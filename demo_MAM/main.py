import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from data.data_generation import *
from optimization import *
import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':
# =============================================================================
# Data GenerationA
#     For robust regression:
#         The parameter noise_type :
#           "None" "Gaussian"  "mean"  "modal" "studentT"  "chiSquare" "mixGauss" 
#      ** where "mean"  "modal" "studentT"  are present in the paper (\epsilon^A,\epsilon^B,\epsilon^C), respectively
#     For classification:
#               Robust Classification
#         generate_corrupted_classification(number=2000,dimension=100,percentage=0.1)
#               Imbalanced classification
#         generate_imbalanced_classification(number=2000,dimension=100,ratio=0.15) 
#               Multi-objective learning for both robust and imbalanced classification
#         generate_multi_classification(number=2000,dimension=100,ratio=0.15) 
# =============================================================================
    # train_loader, validation_loader, testX, testY  = generate_regression(number=2000, dimension=100,  noise_type='modal')
        
        
# =============================================================================
#         Run MAM algorithm
# =============================================================================
    # vnet=Meta_Additive_models(train_loader, validation_loader, testX, testY,total_dimension=100*3,task='regression')
    
# =============================================================================
# For Classification tasks
# =============================================================================
    train_loader, validation_loader, testX, testY = generate_imbalanced_classification(number=100,dimension=10,ratio=0.15)
        # = generate_corrupted_classification(number=2000,dimension=100,percentage=0.3)
    #     # = generate_imbalanced_classification(number=2000,dimension=100,ratio=0.15)
    #     # = generate_multi_classification(number=2000,dimension=100,ratio=0.15)
        
    vnet=Meta_Additive_models(train_loader, validation_loader, testX, testY,total_dimension=10*5,task='classification')
    
    
