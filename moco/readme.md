# MoCo and MoCo v2 implementation

1. Change from the given implementation of MoCo:
  - assume only use one GPU, removed all code and arg parse in code.
  - support one gpu process first, then multiple gpu
    - remove all concate_all_gather
2. improvements
  - one processing one batch of 256 samples, COLAB memory excceeded
    - using 128 sample
  - add validation code
  - print loss at each epoch
  - remove top n evaluation metric
3. unsure improvements:
  - remove crop data argumentation, and avoid crop to get different input for key and query, significantly faster but might cause low performance
  - should not use original code, positive sample should from the same category. result in 2 implementations
  - TODO: is pinned memory in dataloader necessary?
  - TODO: preload queue with samples (did paper mentioned the operation?)

copyright remove?
find follow-up work method?