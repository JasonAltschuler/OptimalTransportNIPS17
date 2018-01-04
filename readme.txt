##########################################################
# Readme file for MATLAB code for NIPS 2017 paper:
# "Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration"
# -- Jason Altschuler, Jonathan Weed, Philippe Rigollet
##########################################################

Each MATLAB file has documentation inside of it. Here's a high-level overview.

-- NB: our focus writing this code was clarity over speed.
Much further optimization can certainly be done.

-- NB: all references to our "paper" are to the revised NIPS 2017 version, i.e.
"https://papers.nips.cc/paper/6792-near-linear-time-approximation-algorithms
-for-optimal-transport-via-sinkhorn-iteration"

##########################################################

ALGORITHMS:
-- sinkhorn.m:        Implementation of classical Sinkhorn algorithm for matrix scaling.
-- greenkhorn.m:      Implementation of our new Greenkhorn algorithm for matrix scalng.
-- compute_ot_lp.m:   Compute optimal transport directly using a MATLAB linear program solver.
-- round_transpoly.m: Implementation of our algorithm for rounding to transport polytope. 

##########################################################

PLOTTING SCRIPTS:
-- plot_matrixscaling.m: (Figure 2 in paper)
      Compares Sinkhorn vs Greenkhorn for matrix-scaling.

-- plot_varyingforeground.m: (LHS of Figure 3 in paper)
      Compares Sinkhorn vs Greenkhorn for matrix-scaling, when amount of "salient" data is varied.

-- plot_ot_smalleta.m: (RHS of Figure 3 in paper)
      Compares Sinkhorn vs Greenkhorn-based OT with small regularization parameter eta. 

-- plot_ot_gcpb.m: (Figure 1 in supplement of paper)
      Compares Greenkhorn vs algorithm from [GCPB '16] 

##########################################################

INPUT-GENERATION: all in 'input_generation/' sub-directory. Contains code
creating OT instances out of images (either synthetically generated or MNIST).