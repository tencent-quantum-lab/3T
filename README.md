## Dependency

Run the following command from the current directory to build dependency:

```
conda create --name 3T python=3.7
conda activate 3T
pip install -r requirements.txt
```

In addition, you need to have GROMACS installed.
This particular work was done using GROMACS 2021.3, which can be found here:
    https://manual.gromacs.org/documentation/2021.3/download.html

Follow the GROMACS installation instructions here:
    https://manual.gromacs.org/5.1.2/install-guide/index.html

If you have done this correctly, the following command should show you the GROMACS executable we will use in `2_structure_generation/example_3T_workflow.ipynb`:
```
which gmx
```
