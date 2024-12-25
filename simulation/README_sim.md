# Simulation Experiments

Experiment results are stored in the `results` directory as pickle files, and analysis plots and HTN diagrams are stored in the `plots` directory.

## Installation
These experiments are tested on python 3.10. Install python packages using the requirements file, either by running `pip install -r requirements.txt` or `conda install --file requirements.txt`. Then, create a `results` and `plots` folder in the simulation directory.

## Running Experiments
The file `run_sim.py` has two arguments: `--task` specifies whether to run the IKEA chair assembly task (`ikea`) or the drill assembly task (`drill`), and `--verbose` specifies whether to visualize the HTNs of individual tasks. For example, to run the IKEA chair assembly task without visualizing HTNs, run: 
```
python run_sim.py --task ikea --verbose False
```

## Plotting Results
The file `stat_analysis_and_plot_sim.py` can be used to run statistical analysis on and plot the results of the `ikea` and `drill` experiments. The script has one argument: `--task` specifies whether to analyze the results of the IKEA chair assembly task (`ikea`) or the drill assembly task (`drill`). For example, to plot the results of the IKEA chair assembly task, run:
```
python stat_analysis_and_plot_sim.py.py --task ikea
```