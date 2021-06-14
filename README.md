# SurpassingHumans_withDHRL

Code for the paper "Surpassing Human Performance on a Foraging Task with Deep Hierarchical Reinforcement Learning using Human Priors"
Download the zip with data and pretrained models here:
https://figshare.com/s/6b4ab468d85c9c77fddd
save and extract in SurpassingHumans_withDHRL/

### Dependencies
```python 
python 3.7.6
matplotlib 3.2.2
numpy 1.18.5
tensorflow 2.2.0
scikit-learn 0.23.1
```

## GenerateFigures
run GenerateFigures to generate the paper figures
```python
python3 GenerateFigures
```

## Train and Test

Watch out both train and testing may take a while. Testing uses multiprocesses to run the evaluation in parallel over multiple seeds.


```python
python3 Train
```

```python
python3 Test
```