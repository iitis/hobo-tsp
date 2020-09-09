# Space-efficient binary optimization for variational computing

Date: *June 2020*

Person responsible for data: *Adam Glos* (aglos [at] iitis.pl).


The scripts necessary for generating the results provided in the "Space-efficient binary optimization for variational computing" preprint. We follow the notation introduced in the preprint.


## Used software
* Julia v1.2.0
* Optim v0.21.0
* NPZ
* JLD2
* FileIo
* PyPlot
* Statistics
* LineSearches
* TravelingSalesmanExact
* GLPK
* other modules installed by default

## Methodics

The quantum evolution was emulated. Details of how the expected energy and gradient were calculated can be found in `qaoa_utils.jl`.

For optimization we used L-BFGS algorithm. Details on the optimization can be found in `qaoa_optimizers.jl`.

## Commands used

### Input data generation
Hamilton problem:
```
julia hamilton_generator.jl hamilton 5
```
The command will create the directory `hamilton` and generate Hamiltonians for 3, 4, 5 cities. For both HOBO and QUBO it assumes first city is visited at first time-point. For QUBO it requires thus $(N-1)^2$ qubits, and for HOBO $(N-1)\lceil \log(N)\rceil$ qubits.

TSP problem:
```
julia tsp_generator.jl tsp_3 3 100
julia tsp_generator.jl tsp_4 4 100
julia tsp_generator.jl tsp_5 5 100
```
The first command will create the directory `tsp_3` and generate `100` TSP instances for 3 cities, similar for the rest of lines. For both HOBO and QUBO it assumes first city is visited at first time-point. For QUBO it requires thus $(N-1)^2$ qubits, and for HOBO $(N-1)\lceil \log(N)\rceil$ qubits.

### Experiments

Before running experiments, run `julia` and type following lines
```
julia> include("sparse_generator_loader.jl")

julia> generator(4)

julia> generator(6)

julia> generator(9)

julia> generator(12)

julia> generator(16)

```
These command generates necessary files for energy computation.

Experiment results for Hamilton problem can be generated using following command
```
julia hamilton_qaoa_experiment.jl 24 100 15
```
where `24` is the number of cores used, `100` is the number of experiments made, and `15` is the maximal level considered.

Experiment results for TSP problem can be generated using following commands
```
julia tsp_qaoa_experiment.jl tsp_3 100 15 40 24
julia tsp_qaoa_experiment.jl tsp_4 100 15 40 24
julia tsp_qaoa_experiment-splitted.jl tsp_5 100 10 40 24 1
julia tsp_qaoa_experiment-splitted.jl tsp_5 100 10 40 24 2
julia tsp_qaoa_experiment-splitted.jl tsp_5 100 10 40 24 3
julia tsp_qaoa_experiment-splitted.jl tsp_5 100 10 40 24 4
```
where `tsp_[n]` is the directory with TSP instances, `100` is the number of TSP instances, `15` is the maximal level, `40` is the required number of successful optimization runs (trajectories) for each instance and each level, `24` is the number of cores used. The last number for `tsp_5` cases indicates which input data are considered within the run, which enables simpler parallelization on PL-GRID infrastructure.

### Plotting data

Plots were generated with following commands
```
julia hamilton_qaoa_plot_trajectories.jl generate plot
julia tsp_qaoa_plot.jl generate plot

```