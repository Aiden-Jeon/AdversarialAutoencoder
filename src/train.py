import argparse
import pytorch_lightning as pl
from aae import AdversarialAutoEncoder, MNISTDataModule


def main(sample_latent):
    aae = AdversarialAutoEncoder(sample_latent=sample_latent)
    trainer = pl.Trainer(max_epochs=100)
    mnist = MNISTDataModule(batch_size=256, num_workers=8)
    trainer.fit(aae, mnist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_latent",
        type=str,
        default="gaussian_mixture",
        choices=["gaussian_mixture", "swiss_roll"],
    )
    args = parser.parse_args()
    main(sample_latent=args.sample_latent)
