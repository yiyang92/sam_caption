### Semantic-attribute modulation, caption experiments

Idea if semantic attribute modulation for language modelling is based on the paper: https://arxiv.org/abs/1707.00117

- using this model having for image captioning having some challenges as unavaliability of the proper data or lack of correspondence between attributes and caption
- In experiments used lookbook.nu dataset, which was introduced in https://arxiv.org/pdf/1801.10300.pdf
- Script contains data preprocessing script, NIC and attention baselines, as well as SAM experimental model
- Resnet checkpoints can be downloaded from tensorflow models/official/Resnet-50 checkpoint