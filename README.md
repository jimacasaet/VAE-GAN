# VAE-GAN
An implementation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/pdf/1512.09300.pdf) using Keras. 
John Rufino Macasaet, Martin Roy Nabus (CoE 197-Z/EE 298)

## Architecture
### Encoder
![Encoder](https://s3-ap-southeast-1.amazonaws.com/celebadataset/vae_cnn_encoder.png)
### Decoder
![Decoder](https://s3-ap-southeast-1.amazonaws.com/celebadataset/vae_cnn_decoder.png)
### Discriminator
![Discriminator](https://s3-ap-southeast-1.amazonaws.com/celebadataset/vae_cnn_discriminator.png)
### Generator Model
![Generator](https://s3-ap-southeast-1.amazonaws.com/celebadataset/vae_cnn_genmodel.png)
### Encoder and Decoder Trainer (VAE)
![Encoder and Decoder Trainer (VAE)](https://s3-ap-southeast-1.amazonaws.com/celebadataset/vae_cnn_vaemodel.png)

## Syntax
### Training
To start the training:
```
python3 vae_gan.py
```
To resume training from a previous training instance:
```
python3 vae_gan.py continue
```

### Testing
To test the functionality of the model:
```
python3 vae_gan.py test
```

### Generation
```
python3 vae_gan.py generate
```

## Results


## Recommendations and Pitfalls
1. While training on the deep learning machines provided, we noticed that the machines were not configured to use `tensorflow-gpu` (i.e. it cannot access NVIDIA cuDNN).
