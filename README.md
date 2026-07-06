# CIDKT

**Curvature Information Driven Dual Knowledge Transfer for Evolutionary Many Task Optimization**

This repository provides the MATLAB implementation of **CIDKT**, a curvature information driven evolutionary many task optimization algorithm for many task optimization problems.

CIDKT is designed around three tightly coupled components:

1. **Task similarity analysis** via Hessian matrix based curvature information.  
   The local Hessian matrix is estimated around elite solutions, and the similarity between tasks is measured by comparing their dominant curvature directions.

2. **Single Source Elite Immigration strategy (SSEI)**.  
   SSEI combines the Euclidean transfer direction and the dominant curvature direction to generate curvature guided elite offspring for the target task.

3. **Multi Source Gaussian Model guided Collaborative Evolution strategy (MGMCE)**.  
   MGMCE selects multiple related auxiliary tasks, extracts their curvature directions, and adaptively adjusts the mean or covariance of the target task Gaussian model according to direction consistency.

The paper reports that CIDKT achieves competitive performance on the **CEC19**, **WCCI20**, **LSMaTSO** and **STOP**  many task benchmark suites, and shows strong effectiveness in terms of AFV, MSS, Wilcoxon rank-sum test, Friedman mean rank, Holm post hoc test, and effect size metrics.

# How to Use

## 1. Install the MTO Platform

First, download and configure the **MTO Platform (MToP)**:

https://github.com/intLyc/MTO-Platform

CIDKT is implemented based on the algorithm interface of MToP. Therefore, MToP should be correctly installed before running the code.

## 2. Add the algorithm file

Copy the following file into the corresponding algorithm folder of the MTO Platform:

- `CIDKT.m`
  

Please make sure that the file name is consistent with the class name in the MATLAB code. If you rename the algorithm file, the class name should also be modified accordingly.

## 3. Run CIDKT in the platform

You can run CIDKT from the MTO Platform GUI or command line by selecting the following algorithm name:

```matlab
mto('CIDKT','WCCI20_MaTSO1',30,true,50,false,'CIDKT_WCCI20SO1');
```

## 4. Recommended experimental setting

```matlab
Population size: 100
Maximum function evaluations: 5 × 10^6
```

## 5. Main parameters in the released code

The default parameter settings in the released code are:

```matlab
MuC      = 2;
MuM      = 5;
KTN      = 5;
rho0   = 0.5;
Gap      = 5;
delta0  = 0.5;
ParaMin  = 0.05;
ParaMax  = 0.95;
vartheta = 0.35;
theta    = 0.5;
d_star   = 10;
```

The meanings of these parameters are listed below:

- `MuC`: distribution index of simulated binary crossover
- `MuM`: distribution index of polynomial mutation
- `KTN`: number of selected auxiliary tasks for knowledge transfer
- `rho0`: initial probability related to transfer strategy selection
- `Gap`: parameter update interval
- `delta0`: initial probability of intra task evolution
- `ParaMin`: lower bound of adaptive parameters
- `ParaMax`: upper bound of adaptive parameters
- `vartheta`: direction consistency threshold
- `theta`: strength of knowledge transfer
- `d_star`: number of dominant curvature directions retained for similarity calculation

