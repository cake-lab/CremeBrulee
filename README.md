# Many Models at the Edge: Characterizing and Improving Deep Inference via Model-Level Caching


This repository contains the code for the paper Many Models at the Edge: Characterizing and Improving Deep Inference via Model-Level Caching, presented at [ACSOS'21](https://conf.researchr.org/home/acsos-2021).

## Overview

As deep learning models becoming more widely used the main limiting resource will become memory.
This drives a need for the development of techniques for conserving memory.
We propose _model-level caching_ for this.

We first motivate the need for model-level caching by examining the scale of the workload expected for deep learning and the needs of models.
When considered together it is apparent that while a few models are very popular, and thus would use dedicated host, the over 97% of models sit idle more than half the time.
This implies that they are more constrained by memory availability than they are computational.

Addressing this will require prioritizing the usage of memory through _model-level caching_.
We introduce a first step in this direction, a simple eviction policy for a model cache that can free up space for new models.
This policy is called CremeBrulee and we realize it in a testbed and in simulation.


## Structure


- `src-simulator`
  - Code for simuation
- `src-testbed`
  - Code for simulation
- `src-workload`
  - Code for generation event drive simulation
- `run_scripts`
  - Scripts used to generate workload and run testbed and simulation
- `datasets`
  - Location for azure FaaS datasets to be store, as well as instructions on finding them
- `docker`
  - Contains custom docker files for testbed implementation
- `measurements`
  - Basic model measurements for deep learning models from AWS `m5.large` instance for simulation.
- `models`
  - Location to keep models, including script to create sample models from Keras


## Citation

```
@INPROCEEDINGS{Ogden2021b,
  author={Ogden, Samuel S. and Gilman, Guin and Walls, Robert J. and Guo, Tian},
  booktitle={2021 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS)}, 
  title={Many Models at the Edge: Characterizing and Improving Deep Inference via Model-Level Caching}, 
  year={2021}
}
```

## Acknowledgements

This work is supported in part by National Science Foundation's support under grants CNS-#1755659 and CNS-#1815619.
Additionally we would like to thank [Scott Gubrud](https://github.com/GubrudScott) for his contributions to an early iteration of the code base.
Finally, we would like to thank all anonymous reviewers for their insightful feedback which helped improve this paper.

