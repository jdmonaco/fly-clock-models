# Source code: Tabuchi, Monaco, et al. (2018)

This repository contains all the research code used to implement
the statistical timing models, synthetic generation of optogenetic
spike-trains, and biophysical clock-neuron models presented in this paper
([link](https://doi.org/10.1016/j.cell.2018.09.016)).

> Tabuchi M, Monaco JD, Duan G, Bell BJ, Liu S, Zhang K, and Wu MN. (2018).
> Clock-generated temporal codes determine synaptic plasticity to control sleep.
> Cell, 175(5), 1213–27. doi: 10.1016/j.cell.2018.09.016

### Summary

> Neurons use two main schemes to encode information: rate coding (frequency of
> firing) and temporal coding (timing or pattern of firing). While the importance
> of rate coding is well established, it remains controversial whether temporal
> codes alone are sufficient for controlling behavior. Moreover, the molecular
> mechanisms underlying the generation of specific temporal codes are enigmatic.
> Here, we show in Drosophila clock neurons that distinct temporal spike patterns,
> dissociated from changes in firing rate, encode time-dependent arousal and
> regulate sleep. From a large-scale genetic screen, we identify the molecular
> pathways mediating the circadian-dependent changes in ionic flux and spike
> morphology that rhythmically modulate spike timing. Remarkably, the daytime
> spiking pattern alone is sufficient to drive plasticity in downstream arousal
> neurons, leading to increased firing of these cells. These findings demonstrate
> a causal role for temporal coding in behavior and define a form of synaptic
> plasticity triggered solely by temporal spike patterns.

## Directories

The following directories are added to the parent project directory. They
should be mostly self-explanatory. The main code library is in the `flymodel`
subdirectory.
