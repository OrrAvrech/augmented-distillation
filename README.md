# augmented-distillation
<p align="center">
  <img src="./figures/gibbs_distributions.PNG">
</p>

## Abstract
Model distillation of complex but accurate model ensembles into fast and simple models can benefit from a data augmentation strategy that aims to reduce the teacher-student error. 
FAST-DAD [1] suggested a data augmentation scheme, specific for tabular data that uses Gibbs sampling from a self-attention pseudolikelihood estimator. 
Instead of drawing samples from the full joint distribution, we suggest a simpler and faster approach- sampling directly from the conditionals estimates. In addition, we introduce the use of prediction sets to obtain reliable augmented samples for distillation. 
We observe that our proposed sampling scheme, is slightly better but is much faster and simpler with the potential of further improving performance by feature-specific augmentations.
