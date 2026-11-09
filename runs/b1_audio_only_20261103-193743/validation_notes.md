# B1 Runtime Checks

- Device: NVIDIA GeForce RTX 4080 SUPER; PyTorch 2.14.1+cu130.
- AMP batch-size check: batch 16 completed forward, CTC loss, backward, and optimizer step. Peak allocated memory: 159.1 MiB; peak reserved: 210.0 MiB. No gradient accumulation needed.
- Eight-utterance overfit check: fixed clean training examples, AMP enabled, dropout and weight decay disabled, learning rate 1e-3. Passed at step 290 with CTC loss 0.045800 and CER 0.000000.
- Full-data epoch timing: epoch 0 took 80 s; epoch 1 took 73 s; mean 76.5 s/epoch.