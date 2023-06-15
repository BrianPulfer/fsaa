# FSAA: Feature Space Adversarial Attacks
FSAA allows to create adversarial examples that corrupt the features of the victim models. Various attacks are possible, by altering initialization and update strategies, losses and perceptual masks.
___


## Installation
If you would like to use `fsaa` in you project, simply run:
```bash
pip install fsaa
```
___

## Usage example
```python
from fsaa.attack import attack
from fsaa.utils import (
  get_initializer,
  get_updater,
  get_loss
)

# Model to be attacked
model = MyModel()

# Batch of data (images)
batch = get_batch()

# Label for the attack
features = model(batch)

# Attacking the batch such that features are as
# different as possible in terms of cosine similarity
# but close in LPIPS distance to the original images
adv_batch = attack(
  model,
  batch,
  labels=features,
  steps=1,
  initializer=get_initializer("RandomInitializer"),
  updater=get_updater("PGDUpdate"),
  image_loss=get_loss("LPIPSAlexLoss"),
  feature_loss=get_loss("CosSimLoss"),
  ilw=0.01,
  flw=1,
  device=device
)
```
A more complete example can be found [here](fsaa/examples/tutorial.py).

___
## Contributing
Contributions are highly welcome! Please refer to the [contributing guidelines](./CONTRIBUTING.md).
___
## License
The code is distributed according to the **Attribution-NonCommercial 4.0 International** [LICENSE](./LICENSE).
___

## Citation
If you used this library as part of your work, please cite the repository with an URL to this webpage.
___

## Acknowledgements
Part of the code was taken and adapted from the following repositories:
  - [facebookresearch/active_indexing](https://github.com/facebookresearch/active_indexing)
    - JND masking
