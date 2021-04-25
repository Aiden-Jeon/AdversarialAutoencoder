# AdversarialAutoencoder
Pytorch Lightning implementation of adverserial autoencoder (AAE)
- paper: https://arxiv.org/abs/1511.05644


# Result of code
In paper, figure 4 shows the example latent for MNIST dataset.
It can be a gaussian mixture or swill roll.
![paper-figure-4](/imgs/paper.png)

My code can generate each of latent.
- Gaussian Mixture

![gaussian-mixture](/imgs/gaussian_mixture.gif)

- Swiss roll

![swiss-roll](/imgs/swiss_roll.gif)


# Requirements
- `python==3.8.7`

# How to use
1. Install requirements
```bash
pip install -r requirements.txt
```

2. Run
```bash
python src/train.py 
```
`train.py` has `--sample_latent` argument.  

2.1) Latent with `swiss_roll`
```bash
python src/train.py --sample_latent swiss_roll
```

2.2) Latent with `gaussian_mixture`
```bash
python src/train.py --sample_latent gaussian_mixture
```
