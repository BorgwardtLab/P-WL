# A persistent Weisfeiler Lehman Procedure for Graph Classification
An extension of persistent homology for categorical graph labels and a Weisfeiler-Lehman inspired graph kernel using persistent homology.

![alt text][logo]

[logo]: https://github.com/BorgwardtLab/P-WL/blob/master/assets/flow.png "Persistence Weisfeiler Lehmann Flow"

### Running the Method

**NOTE**: The classification accuracies reported by running this code will not (!) align with the ones reported in the paper, as no grid search is performed. Thus, the folds used for training will be slightly different because the inner cross-validation loop is lacking. This was a deliberate decision in order to present a brief and quick working example for the review process.

#### Examples

To run `PWL-C` on `MUTAG` with 0 WL iterations and p=1, run 
```
$ cd src
$ python main.py -c -n 0 -p 1 data/MUTAG/*.gml -l data/MUTAG/Labels.txt
```

The arguments for all our methods (with `1 WL iteration` and `p=1`) are as follows:

`PWL`: `main.py -n 1 -p 1 data/...`

`PWL-C`: `main.py -c -n 1 -p 1 data/...`

`PWL-UC`: `main.py -u -c -n 1 -p 1 data/...`


