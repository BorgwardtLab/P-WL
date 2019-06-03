# A Persistent Weisfeiler&ndash;Lehman Procedure for Graph Classification

This repository contains code, data sets, experiments, and documentation
for our ICML 2019 paper
&ldquo;[A Persistent Weisfeiler&ndash;Lehman Procedure for Graph Classification](http://proceedings.mlr.press/v97/rieck19a/rieck19a.pdf)&rdquo;.

![Workflow of the persistent Weisfeiler--Lehman procedure][logo]

[logo]: https://github.com/BorgwardtLab/P-WL/blob/master/assets/flow.png "Workflow of the persitent Weisfeiler--Lehman procedure"

## Running the Method

**Warning**: The classification accuracies reported by running this code
will not&nbsp;(!) align with the ones reported in the paper, as no grid
search is performed. Thus, the folds used for training will be slightly
different because there is no inner cross-validation loop.

## Examples

To run `PWL-C` on `MUTAG` with 0 WL iterations and p=1, run 
```
$ cd src
$ python main.py -c -n 0 -p 1 data/MUTAG/*.gml -l data/MUTAG/Labels.txt
```

The arguments for all our methods (with `1 WL iteration` and `p=1`) are as follows:

`PWL`: `main.py -n 1 -p 1 data/...`

`PWL-C`: `main.py -c -n 1 -p 1 data/...`

`PWL-UC`: `main.py -u -c -n 1 -p 1 data/...`

## Results

These are the results reported in the paper. For convenience reasons, we
summarise them here. This table will be updated as soon as other baselines
are available.

|             | D & D        | MUTAG        | NCI1         | NCI109       | PROTEINS     | PTC-MR       | PTC-FR       | PTC-MM       | PTC-FM       | 
|-------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------| 
| V-Hist      | 78.32 ± 0.35 | 85.96 ± 0.27 | 64.40 ± 0.07 | 63.25 ± 0.12 | 72.33 ± 0.32 | 58.31 ± 0.27 | **68.13 ± 0.23** | 66.96 ± 0.51 | 57.91 ± 0.83 | 
| E-Hist      | 72.90 ± 0.48 | 85.69 ± 0.46 | 63.66 ± 0.11 | 63.27 ± 0.07 | 72.14 ± 0.39 | 55.82 ± 0.00 | 65.53 ± 0.00 | 61.61 ± 0.00 | 59.03 ± 0.00 | 
| RETGK   | **81.60 ± 0.30** | 90.30 ± 1.10 | 84.50 ± 0.20 |              | 75.80 ± 0.60 | 62.15 ± 1.60 | 67.80 ± 1.10 | 67.90 ± 1.40 | 63.90 ± 1.30 | 
| WL          | 79.45 ± 0.38 | 87.26 ± 1.42 | 85.58 ± 0.15 | 84.85 ± 0.19 | **76.11 ± 0.64** | 63.12 ± 1.44 | 67.64 ± 0.74 | 67.28 ± 0.97 | 64.80 ± 0.85 | 
| DEEP-WL |              | 82.94 ± 2.68 | 80.31 ± 0.46 | 80.32 ± 0.33 | 75.68 ± 0.54 | 60.08 ± 2.55 |              |              |              | 
| P-WL        | 79.34 ± 0.46 | 86.10 ± 1.37 | 85.34 ± 0.14 | 84.78 ± 0.15 | 75.31 ± 0.73 | 63.07 ± 1.68 | 67.30 ± 1.50 | **68.40 ± 1.17** | 64.47 ± 1.84 | 
| P-WL-C      | 78.66 ± 0.32 | **90.51 ± 1.34** | 85.46 ± 0.16 | **84.96 ± 0.34** | 75.27 ± 0.38 | **64.02 ± 0.82** | 67.15 ± 1.09 | **68.57 ± 1.76** | **65.78 ± 1.22** | 
| P-WL-UC     | 78.50 ± 0.41 | 85.17 ± 0.29 | **85.62 ± 0.27** | **85.11 ± 0.30** | 75.86 ± 0.78 | **63.46 ± 1.58**| 67.02 ± 1.29 | **68.01 ± 1.04** | **65.44 ± 1.18** | 

## Additional experiments

The repository contains additional experiments, which will be documented
and extended over time:

- Accuracy per iteration step of `WL` vs. `P-WL` vs. `P-WL-C`
- Feature importance of cycle features (forthcoming)
- Kullback&ndash;Leibler divergence and Jensen&ndash;Shannon divergence
- Comparison with additional baselines and graph kernels&nbsp;(see also
  the [GraphKernels repository](https://github.com/BorgwardtLab/GraphKernels)
  for more of them)

## Help

If you have questions concerning P-WL or you encounter problems when
trying to build the tool under your own system, please open an issue in
[the issue tracker](https://github.com/BorgwardtLab/P-WL/issues). Try to
describe the issue in sufficient detail in order to make it possible for
us to help you.

## Contributors

P-WL is developed and maintained by members of the [Machine Learning and
Computational Biology Lab](https://www.bsse.ethz.ch/mlcb) of [Prof. Dr.
Karsten Borgwardt](https://www.bsse.ethz.ch/mlcb/karsten.html):

- Bastian Rieck ([GitHub](https://github.com/Submanifold))
- Christian Bock ([GitHub](https://github.com/chrisby))

## Citation 

Please use the following BibTeX citation when using our method or
comparing against it:

    @InProceedings{Rieck19b,
      title     = {A Persistent {W}eisfeiler--{L}ehman Procedure for Graph Classification},
      author    = {Rieck, Bastian and Bock, Christian and Borgwardt, Karsten},
      booktitle = {Proceedings of the 36th International Conference on Machine Learning},
      pages     = {5448--5458},
      year      = {2019},
      editor    = {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
      volume    = {97},
      series    = {Proceedings of Machine Learning Research},
      address   = {Long Beach, California, USA},
      month     = jun,
      publisher = {PMLR},
      pdf       = {http://proceedings.mlr.press/v97/rieck19a/rieck19a.pdf},
      url       = {http://proceedings.mlr.press/v97/rieck19a.html},
      abstract  = {The Weisfeiler–-Lehman graph kernel exhibits competitive performance in many graph classification tasks. However, its subtree features are not able to capture connected components and cycles, topological features known for characterising graphs. To extract such features, we leverage propagated node label information and transform unweighted graphs into metric ones. This permits us to augment the subtree features with topological information obtained using persistent homology, a concept from topological data analysis. Our method, which we formalise as a generalisation of Weisfeiler-–Lehman subtree features, exhibits favourable classification accuracy and its improvements in predictive performance are mainly driven by including cycle information.}
    }
