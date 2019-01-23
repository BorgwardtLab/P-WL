## "A Persistent Weisfeiler-Lehmann Procedure for Graph Classification" submitted to ICML 2019
### Installation Instructions for accompanying code

In this docker container we provide code to run our method on five data sets (`MUTAG`, `PTC_MR`, `PTC_MM`, `PTC_FR`, `PTV_FM`). 

### Docker Installation
To run the method you need to install Docker. You can download the installation file for your OS [here](https://www.docker.com/get-started)
If you do not have a docker account or you don't want to create one, you can use this [link](https://download.docker.com).

### Building the Container
To build the docker container open a command line and navigate to the folder where this `README.me` files lies and run the following command:
```
$ docker build -t pwl .
```

This might take a while as all dependencies are now being built. 

### Running the Method

Once you see a message ``, the container was sucessfully built. 

`docker exec -it pwl-container python main.py -c -n 0 -p 1 data/MUTAG/*.gml -l data/MUTAG/Labels.txt` 
