## Tiered Tensor Transform

This repository demonstrates the workflow and algorithm of Tencent Quantum Lab's Tiered Tensor Transform (3T). This is a multi-scale structure optimization/generation algorithm, which relies on one initial structure example to generate multiple conformations of protein-ligand complex pockets. This task is traditionally difficult to do, and the conventional approach is to generate ligand docking poses on rigid protein pockets. Either that, or run a full-blown molecular dynamics (MD simulation) on the full protein-ligand complex pocket to generate the desired conformations (but this is not suitable for large scale ligand screening commonly used in computational drug discovery). On the other hand, 3T utilizes local transformations to hierarchically transform the initial structure, generating new structures in the process while easily escaping trivial local energy traps which is usually very difficult to escape without lengthy MD.

![Alt text](2_structure_generation/Images/CDK2_1fin_2iw8-4SP.png?raw=true "Title")

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
    https://manual.gromacs.org/2021.3/install-guide/index.html

If you have done this correctly, the following command should show you the GROMACS executable we will use in `2_structure_generation/example_3T_workflow.ipynb`:
```
which gmx
```
