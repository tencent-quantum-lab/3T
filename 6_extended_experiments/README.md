## Aggressive 3T

This example demonstrates the workflow for an aggressive 3T struture generation process, as opposed to the gentle version used in the majority of the manuscript and folder `2_structure_generation`.
In the aggressive 3T, we start with strong initial kick followed by very aggressive energy minimization (10k steps). Because of how aggressive this approach is, this step 1 kick is only allowed for the ligand while the protein is frozen.
It is then followed by the step 2 kick, which is a gentle kick involving the ligand and the entire protein pocket (5k steps).
Finally, we further relax the entire pocket structure with step 3 relaxation (5k steps, with no additional energetic kick).

On a single NVidia T4 GPU, step 1 takes ~15 minutes, step 2 takes 35-40 minutes, and step 3 takes 35-40 minutes.

We demonstrate this example on two cases:

### 1. Fix bad initial cross-docked structure, explore flip conformations and significantly improve ligand pose.
In this case, we first cross-dock ligand extracted from 3r9h co-crystal onto CDK2 protein structure extracted from 1fin co-crystal. The original smina cross-docked result produces a completely wrong ligand pose. Using aggressive 3T, we can explore new ligand poses within a fully flexible protein pocket.

Green is the experimental co-crystal structure (3r9h), white is the initial smina cross-docked structure (3r9h on 1fin), and magenta is the aggressive 3T structure (generated from the smina cross-dock initial structure).

![Alt text](data/3r9h_github_figure.png?raw=true "Title")

### 2. Exploring alternative binding modes in nearby pockets:
In this case, we first self-dock ligand extracted from 3nya co-crystal onto ADRB2 protein structure extracted from 3nya co-crystal. The original smina self-docked result still sits in the original orthosteric binding site. Using aggressive 3T, we can explore ligand poses within the nearby extracellular vestibule binding site.

White is the experimental co-crystal structure (3nya) and green is the aggressive 3T structure (generated from the smina self-dock initial structure).

![Alt text](data/3nya_github_figure.png?raw=true "Title")

Compared to the main text, the primary algorithm code modifications are located in `minimize_GL_complex.py` and `GL_modules/GL_potential_model.py`.

## Usage

Run the Jupyter notebooks.

Data preprocessing & postprocesing examples:

```
jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True 6-extended_experiments.ipynb
```

Structure generation examples.  Make sure to check `2_structure_generation` for pre-requisites:

```
jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True 6-aggressive_3T.ipynb
```
