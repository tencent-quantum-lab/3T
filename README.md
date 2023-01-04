## Tiered Tensor Transform

This repository demonstrates the workflow and algorithm of Tencent Quantum Lab's Tiered Tensor Transform (3T).
This is a multi-scale structure optimization/generation algorithm, which relies on one initial structure example to generate multiple conformations of protein-ligand complex pockets.
This task is traditionally difficult to do, and the conventional approach is to generate ligand docking poses on rigid protein pockets.
Either that, or run a full-blown molecular dynamics (MD) simulation on the full protein-ligand complex pocket to generate the desired conformations (but this is not suitable for large scale ligand screening commonly used in computational drug discovery).

On the other hand, 3T utilizes tensor-based local structure transformations to hierarchically transform the initial structure,
generating new ligand-dependent protein-ligand complex pocket structures in the process while easily escaping trivial local energy traps
which are usually very difficult to escape from without a long MD:

![Alt text](2_structure_generation/Images/3T_Model.png?raw=true "Title")

As can be seen from the figure above, 3T does not require training data. The structure transformation module (bottom) is purely about geometry transformation,
while the structure evaluation cost function (top) is just a differentiable classical force field function.
Because of this, no machine learning (ML) training data is necessary for 3T structure generation.
Running backward propagation on this 3T model does not directly update the structure;
it updates the local structure translation and rotation transformation parameters instead.

3T is neither an ML nor an MD approach, but it does utilize many components found in both ML and MD.

In order to generate multiple different conformations, we distort the initial structure with a random energetic kick,
before relaxing the structure back to generate the final protein-ligand conformations, as shown in the figure below:

![Alt text](2_structure_generation/Images/3T_Workflow.png?raw=true "Title")

The corresponding manuscript describing this work is posted on arXiv: https://arxiv.org/abs/2301.00984.

## Dependency

The code was tested on Linux operating system. Run the following command from the current directory to install most of the dependencies:

```
conda create --name 3T python=3.7
conda activate 3T
conda install --file requirements.txt -c pytorch -c conda-forge -c rdkit -c openbabel -c anaconda
```

The conda library versions in `requirements.txt` are provided as reference only. If your conda has difficulty in resolving the dependencies simultaneously, you can either install them one by one or simply remove the suggested library version constraints from `requirements.txt`.

In addition, you need to have GROMACS installed. You may need to first install cmake before installing GROMACS.
This particular work was done using GROMACS 2021.3 (<b><ins><i>other versions may work as well, but have not been tested</i></ins></b>). Download link and installation instructions are available here: <br />
&ensp;https://manual.gromacs.org/documentation/2021.3/download.html <br />
&ensp;https://manual.gromacs.org/2021.3/install-guide/index.html

Or if you'd like, you can try this convenient installation option (<b>not recommended</b>): <br />
`conda install -c bioconda gromacs`

A separate library crucial for the 3T structure generation (a custom version of the [InterMol](https://github.com/shirtsgroup/InterMol) library) will be automatically installed when running the `2_structure_generation/02_generation_workflow.ipynb` notebook.

## Contact

Questions about this repository may be addressed to Jonathan Mailoa ( jpmailoa [AT] alum [DOT] mit [DOT] edu ).
