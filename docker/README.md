## "A Persistent Weisfeiler-Lehmann Procedure for Graph Classification" submitted to ICML 2019
### Installation Instructions for accompanying code

In this docker container we provide code to run our method on five data sets (`MUTAG`, `PTC_MR`, `PTC_MM`, `PTC_FR`, `PTV_FM`). 

### Docker Installation
To run the method you need to install Docker. You can download the installation file for your OS [here](https://www.docker.com/get-started).
If you do not have a docker account or you don't want to create one, you can use this [link](https://download.docker.com).

### Building the Container
To build the docker container open a command line and navigate to the folder where this `README.md` files lies and run the following command:
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
*NOTE*: The performances reported by running this code will now align with the ones reported in the paper, as no grid search is performed.
#### Examples
To run `PWL-C` on `MUTAG` run with 0 WL iterations and p=1, run 
```
docker exec -it pwl-container python main.py -c -n 0 -p 1 data/MUTAG/*.gml -l data/MUTAG/Labels.txt
```

The main script allows for the following arguments:
```
usage: main.py [-h] [-d DATASET] -l LABELS [-n NUM_ITERATIONS] [-c] [-u]
               [-p POWER]
               FILES [FILES ...]

positional arguments:
  FILES                 Input graphs (in some supported format)

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Name of data set
  -l LABELS, --labels LABELS
                        Labels file
  -n NUM_ITERATIONS, --num-iterations NUM_ITERATIONS
                        Number of Weisfeiler-Lehman iterations
  -c, --use-cycle-persistence
                        Indicates whether cycle persistence should be
                        calculated or not
  -u, --use-uniform-metric
                        Use uniform metric for weight assignment
  -p POWER, --power POWER
                        Power parameter for metric calculations
```

