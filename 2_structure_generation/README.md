## Dependency

Make sure you have GROMACS (`gmx`), `wget`, and `unzip` installed. The notebook will first check that they are properly installed before it can run.
If you have performed the installations correctly, the command `which gmx`, `which wget`, and `which unzip` should show you the corresponding executable locations.
If they're not installed correctly, you can do the installation using these instructions: <br />
&ensp;https://manual.gromacs.org/documentation/2021.3/download.html <br />
&ensp;https://manual.gromacs.org/2021.3/install-guide/index.html <br />
&ensp;https://linuxize.com/post/wget-command-examples/ <br />
&ensp;https://linuxpip.org/install-zip-unzip/

## Evaluation

To check the output we have pre-generated for you, simply open the notebook:

```
jupyter notebook 02_generation_workflow.ipynb
```


## Run

If you would like to re-run the notebook on your machine, make sure that the dependencies (check `../README.md`) are installed.
Then simply run or modify the notebook interactively:

```
jupyter notebook 02_generation_workflow.ipynb
```

Otherwise, you can also just run all the cells in the notebook as-is in your machine directly using the following command, and wait. It will take less than 1 hour on a GPU machine, and less than 10 hours on a single-core CPU machine:

```
jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True 02_generation_workflow.ipynb
```
