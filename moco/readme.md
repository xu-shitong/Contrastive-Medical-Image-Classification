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
  - visualize training loss
  - TODO: implement t-sne
3. unsure improvements:
  - remove crop data argumentation, and avoid crop to get different input for key and query, significantly faster but might cause low performance
  - should not use original code, positive sample should from the same category. result in 2 implementations
    - NEED GUIDE: multi-label loss result in high loss, might be difficult to compare
    - TODO: validation loss needs reviewing, too close to the training loss, maybe should not compare to the queue from training sample 
    - TODO: use BCE???
  - TODO: is pinned memory in dataloader necessary?
  - TODO: preload queue with samples (did paper mentioned the operation?)

todo:
- try uncoloured mlp classify
- try run whole program, using moco and non-moco
- train supervised baseline
- gradcam


thesis:
- add medical itself prior work


