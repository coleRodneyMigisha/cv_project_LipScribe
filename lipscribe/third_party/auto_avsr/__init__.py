"""Visual front-end and Conformer encoder vendored from Auto-AVSR (Apache 2.0, see LICENSE).

Source: https://github.com/mpc001/auto_avsr at commit 182b628 (espnet/nets/pytorch_backend).
Only the modules needed by the visual encoder are copied: frontend/resnet.py,
encoder/conformer_encoder.py, nets_utils.py and transformer/{attention, embedding,
layer_norm, positionwise_feed_forward, repeat}.py. Their package imports were changed to
relative imports; nothing else was modified. LipScribe uses these classes only as the
starting point for its own fine-tuning (lipscribe.models.pretrained).
"""
