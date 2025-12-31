# Reproduction of "Meta Additive Model: Learning Theory Analysis and Applications" 

## Setups

The main requiring environment is as bellow:  

- Python 3.10
- Torch 2.0.1 
- CUDA 11.7

## Run Meta Additive Model  （run main.py）

#### （1）Generate synthetic data for specific tasks:
```python
train_loader, validation_loader, testX, testY = generate_regression(number=2000, dimension=100,  noise_type='mean')
```


##### *The parameter noise_type :*
```python
*"None" "Gaussian"  "mean"  "modal" "studentT"  "chiSquare" "mixGauss"* 

*** where "mean"  "modal" "studentT"  are present in the paper (\epsilon^A$,\epsilon^B,\epsilon^C), respectively*
```


#### （2）Start to run the MAM optimization procedure

```python
vnet=Meta_Additive_models(train_loader, validation_loader, testX, testY,total_dimension=100*3,task='regression')
```


Similar process is also available for classification tasks.


```python
train_loader, validation_loader, testX, testY = generate_multi_classification(number=1000,dimension=100,ratio=0.15)
    # = generate_corrupted_classification(number=1000,dimension=100,percentage=0.3)
    # = generate_imbalanced_classification(number=1000,dimension=100,ratio=0.15)
    # = generate_multi_classification(number=1000,dimension=100,ratio=0.15)
    
vnet=Meta_Additive_models(train_loader, validation_loader, testX, testY,total_dimension=100*3,task='classification')
```
