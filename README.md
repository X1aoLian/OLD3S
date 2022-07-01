# Online Deep Learning from Doubly-Streaming Data
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
## Abstract
This paper investigates a new online learning problem with doubly-streaming data,
where the data streams are described by feature spaces that constantly evolve.
    The challenges of this problem are two folds.
    1) Data instances that flow in ceaselessly
    are not likely to always follow an identical distribution,
    require the learners to be updated on-the-fly.
    2) New features that just emerge are described by 
    very few data instances, 
    result in *weak* learners that tend to make error predictions.
    To overcome,
    a plausible idea is to establish relationship
    between the pre-and-post evolving feature spaces,
    so that an online learner can leverage and adapt 
    the learned knowledge from the old 
    to the new features for better performance.
    Unfortunately, this idea does not scale up to 
    high-dimensional media streams 
    with complex feature interplay,
    suffering an tradeoff between onlineness 
    (biasing shallow learners)
    and expressiveness (requiring deep learners).
    Motivated by this,
    we propose a novel OLD<sup>3</sup>S paradigm,
    where a shared latent subspace is discovered 
    to  summarize information from the old and new feature spaces,
    building intermediate feature mapping relationship.
    A key trait of OLD<sup>3</sup>S is to treat
    the *model capacity* as a learnable semantics,
    yields optimal model depth and parameters jointly in accordance 
    with the complexity and non-linearity of the inputs
    in an online fashion.
    Both theoretical analyses and extensive experiments benchmarked on
    real-world datasets including images and natural languages
    substantiate the viability and effectiveness of our proposal.
## Requirements
This code was tested on Windows and macOS
```
conda create -n OLDS python=3.9
conda activate OLDS
pip install torchvision
pip install pandas
pip install matplotlib
pip install sklearn

```

## Run

```
conda activate OLDS
python main.py -DataName=cifar -beta=0.9 -learningrate=0.01
python main.py -DataName=reuter -FromLanguage=EN -ToLanguage=FR -beta=0.9 -learningrate=0.01
```

