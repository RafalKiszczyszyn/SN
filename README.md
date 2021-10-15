### Labolatorium 1.

Uwaga! Do poprawnego działania programu potrzebna jest biblioteka Numpy.

Aby uruchomić wybrane badanie, otwórz `Lab1/x.py` i odkomentuj odpowiednią linię. Np. żeby uruchomić badanie wsp. uczenia dla adaline należy odkomentować adaline_eta_bipolar:
```python
    # perceptron_threshold_bipolar(X, y, eta=0.01, w_range=0.01)
    # perceptron_threshold_unipolar(X_, y_, eta=0.01, w_range=0.01)
    # perceptron_w_range_bipolar(X, y, eta=0.01)
    # perceptron_w_range_unipolar(X_, y_, eta=0.01)
    # perceptron_eta_bipolar(X, y, w_range=0.3)
    # perceptron_eta_unipolar(X_, y_, w_range=0.3)
    # adaline_w_range_bipolar(X, y, eta=0.01, min_cost=0.5)
    # adaline_w_range_unipolar(X_, y_, eta=0.01, min_cost=0.1)
    adaline_eta_bipolar(X, y, min_cost=0.5, w_range=0.5)
    # adaline_eta_unipolar(X_, y_, min_cost=0.1, w_range=0.5)
    # adaline_min_cost_bipolar(X, y, eta=0.01, w_range=0.5)
    # adaline_min_cost_unipolar(X_, y_, eta=0.01, w_range=0.01)
```

Aby przetestować wybrany model dla różnych wektorów wejściowych należy uruchomić w cmd:
```
python x.py perceptron -1 1
```
lub 
```
python x.py adaline -1 1
```
