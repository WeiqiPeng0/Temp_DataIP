#### UPDATE 3/25:
Pleasee see QuickStart_new notebook.
- fixed testing scaling issue.
- added new negative gradient matching loss w.r.t. x (i.e. clean images) 


Please refer to QuickStart for some starter code.

- matchloss is the gradient matching loss.
- previosly I selected 500 images across classes for training one seed image. the result is somewhat promising.
- consider training seed image for each lass and see if this improves transferability.
- for testing training a new model (w/ different architecture for sure) on the perturbed dataset and do inference with the seed image.
