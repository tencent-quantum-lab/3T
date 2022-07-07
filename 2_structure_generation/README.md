## Evaluation

To check the output we have pre-generated for you, simply open the notebook:

```
jupyter notebook example_3T_workflow.ipynb
```


## Run

If you would like to re-run the notebook on your machine, make sure that the dependencies (check `../README.md`) are installed.
Then simply run or modify the notebook interactively:

```
jupyter notebook example_3T_workflow.ipynb
```

Otherwise, you can also just run all the cells in the notebook as-is in your machine directly using the following command, and wait. It will take less than 1 hour on a GPU machine, and less than 10 hours on a single-core CPU machine:

```
jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True example_3T_workflow.ipynb
```
