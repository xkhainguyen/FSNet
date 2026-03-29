<img src="figure_r1_compare_R3.png" alt="Figure R1" width="900" height="500">

Figure R1. **Results of our approach for a PINN using R3 adaptive sampling on the physics-inspired learning task.**

---
<img src="figure_r2_learning_curves.png" alt="Figure R2">

Figure R2. **SL (left) and SSL (right) dynamics across epochs in the synthetic optimization task.** On the left, the solid curve (left y-axis) shows a U-shaped validation merit trajectory, while the dashed curve (right y-axis) shows the training supervised loss decreasing monotonically. On the right, our SSL, warm-started from the checkpoint with the lowest validation merit (epoch 250), outperforms vanilla SSL and other variants. This highlights the objective mismatch in purely data-driven training and the importance of our merit-based criterion. The total training cost of our method is the sum of "inexact but cheap" SL epochs and "faithful but expensive" SSL epochs.

---
Table R1. **Performance comparison on a synthetic constrained optimization benchmark of SSL warm-started from different SL pretraining steps. SL achieves lowest validation merit at around epoch 250.**

<img src="table_r1_epochs.png" alt="Table R2">

---
Table R2. **Performance comparison on a synthetic constrained optimization benchmark of SSL with no warm-start, random label warm-start and cheap label warm-start.** We construct random solution labels via uniform sampling over a $\pm 3$ range in each dimension, which matches the scale of the solver-generated solutions. We generate $7{,}000$ such samples and perform supervised pretraining as in our standard pipeline. The last SL checkpoint is selected to warm-start SSL as it has lowest training loss and validation merit.

<img src="table_r2_random.png" alt="Table R2">


