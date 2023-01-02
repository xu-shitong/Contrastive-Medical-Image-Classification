# MoCo and MoCo v2 implementation

1. Change from the given implementation of MoCo:
  - assume only use one GPU, removed all code and arg parse in code.
  - added model eval in training, not sure if necessary
  - swapped ordering of eval and optimizer.step 
  - support one gpu process first, then multiple gpu
    - remove all concate_all_gather, test on colab with local builder code
  - TODO: new error saved in ipynb file

1. one processing one batch of 256 samples, COLAB memory excceeded
  - might be able to use 128 sample