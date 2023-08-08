import tensorflow as tf
from config import _C as cfg
from models.models import VAE 
from datasets.agroverse import make_gen


cfg = cfg.clone()


def main():
    print(cfg)
    vae_model = VAE(cfg)
    
    vae_model.compile
    #train_gen, valid_gen, num_train_samples, normalization_params = make_gen(cfg)

if __name__ == "__main__":
    main()