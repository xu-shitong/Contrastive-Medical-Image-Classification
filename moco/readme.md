# MoCo and MoCo v2 implementation

1. Change from the given implementation of MoCo:
  - assume only use one GPU, removed all code and arg parse in code.
  - support one gpu process first, then multiple gpu
    - remove all concate_all_gather
1. improvements
  - one processing one batch of 256 samples, COLAB memory excceeded
    - using 128 sample
  - add validation code
  - print loss at each epoch
  - TODO: is pinned memory in dataloader necessary?
  - TODO: should not use original code, positive sample should from the same category!
  - TODO: might cannot use validation, due to model change its comparison queue