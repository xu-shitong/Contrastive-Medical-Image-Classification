# MoCo and MoCo v2 implementation

1. Change from the given implementation of MoCo:
  - assume only use one GPU, removed all code and arg parse in code.
  - added model eval in training, not sure if necessary
  - swapped ordering of eval and optimizer.step 
  - support one gpu process first, then multiple gpu
    - remove all concate_all_gather, test on colab with local builder code
2. improvements
  - one processing one batch of 256 samples, COLAB memory excceeded
    - using 128 sample
  - add validation code
  - print loss at each epoch
  - TODO: is pinned memory in dataloader necessary?
  - TODO: avoid print in the first batch
  - TODO: should not use original code, positive sample should from the same category!
  - TODO: might cannot use validation, due to model change its comparison queue