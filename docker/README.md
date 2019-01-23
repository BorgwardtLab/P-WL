## "A Persistent Weisfeiler-Lehmann Procedure for Graph Classification" submitted to ICML 2019
### Installation Instructions for accompanying code

In this docker container, we provide code to run our method on five data sets (`MUTAG`, `PTC_MR`, `PTC_MM`, `PTC_FR`, `PTC_FM`). 

### Docker Installation
To run the method, you need to install Docker. You can download the installation file for your OS [here](https://www.docker.com/get-started).
If you do not have a docker account or you don't want to create one, you can use this [link](https://download.docker.com).

### Building the Container
To build the docker container open a terminal (command line) and navigate to the folder where this `README.md` file lies and run the following command:
```
$ docker build -t pwl .
```

This might take a while as all dependencies are now being built and an image tagged as `pwl` is created. 
Once you see the `Successfully tagged pwl:latest` message, you can create and start a container named `pwl-container` via

```
$ docker create -it --name pwl-container pwl
$ docker start pwl-container
```

### Running the Method

**NOTE**: The classification accuracies reported by running this code will not (!) align with the ones reported in the paper, as no grid search is performed. Thus, the folds used for training will be slightly different because the inner cross-validation loop is lacking. This was a deliberate decision in order to present a brief and quick working example for the review process.

#### Examples

To run `PWL-C` on `MUTAG` with 0 WL iterations and p=1, run 
```
$ docker exec -it pwl-container python main.py -c -n 0 \
 -p 1 data/MUTAG/*.gml -l data/MUTAG/Labels.txt
```

Notice that we wrapped this command such that it fits onto the page. You
may need to specify it in one line.

The arguments for all our methods (with `1 WL iteration` and `p=1`) are as follows:

`PWL`: `main.py -n 1 -p 1 data/...`

`PWL-C`: `main.py -c -n 1 -p 1 data/...`

`PWL-UC`: `main.py -u -c -n 1 -p 1 data/...`

