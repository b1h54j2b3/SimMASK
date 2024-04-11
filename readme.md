## Dependencies
* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)==1.7.0

## Training & Evaluation

```
./go.sh 
```
```
python simmask.py $DATASET_NAME $BETA $P

DATASET_NAME is the name of dataset, BETA is controller parameter ranging from 0 to 1, P is the possibility threshold for mask which ranges from 0 to 1.
```