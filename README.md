# Latent SDE MoCap Reproduction Attempt

This repository contains code to attempt reproducing the motion capture experiments from [Li et al.'s work on Scalable Gradients for SDEs](https://arxiv.org/abs/2001.01328). The implementation is based on the [torchsde Lorenz attractor example](https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py), modified to work with motion capture data and incorporating architectural improvements from [Yıldız et al.'s work](https://arxiv.org/abs/1905.10994) on "Deep generative second order ODEs with Bayesian neural networks".

**Current Status**: While we've implemented the architecture and training procedure, our Test MSE (~9.0) has not yet reached the performance reported in the original paper (~4.0). Work is ongoing to improve the results.

## Key Modifications from Original torchsde Example

- Replaced Lorenz attractor data with motion capture dataset as training target
- Implemented architecture from ["Scalable Gradients for Stochastic Differential Equations"](https://arxiv.org/abs/2001.01328) (Li et al., AISTATS 2020)
- Added Weights & Biases logging and model checkpointing
- Adapted training loop for motion capture data

## Dataset

The experiment uses preprocessed motion capture data from the CMU Graphics Lab Motion Capture Database:

- Preprocessed dataset: `mocap35.mat` (walking sequence)
- Original source: [CMU MoCap Database](http://mocap.cs.cmu.edu/)
- Preprocessed data available at: [Google Drive Dataset](https://drive.google.com/drive/folders/1c0UMSqlvZRORmCNN_qVdiqu2n8sKcwnh)

## Setup

1. Download the preprocessed dataset `mocap35.mat`
2. Place the file in the same directory as the code
3. Ensure you have uv installed

## Running the Experiment

To run the experiment:
```bash
uv run latent_sde.py
```

## Project structure
```
.
├── README.md
├── pyproject.toml
├── latent_sde.py
├── .gitignore
├── .python-version
└── mocap35.mat (you need to download this)
```

## License

This code is based on Google LLC's torchsde Lorenz attractor example, which is licensed under the Apache License 2.0. The original code has been modified by Naoki Kiyohara to:
- Use Motion Capture dataset as training target
- Implement architecture from "Scalable Gradients for Stochastic Differential Equations" (Li et al., AISTATS 2020)
- Add Weights & Biases logging and model checkpointing

## Citations

This implementation builds upon several works:

For the original Latent SDE implementation:
```bibtex
@misc{torchsde2020,
    author = {Google Research},
    title = {torchsde},
    year = {2020},
    publisher = {GitHub},
    url = {https://github.com/google-research/torchsde}
}
```

For the Latent SDE model and motion capture experiments ([paper link](https://arxiv.org/abs/2001.01328)):
```bibtex
@inproceedings{li2020scalable,
    title     = {Scalable Gradients and Variational Inference for Stochastic Differential Equations},
    author    = {Li, Xuechen and Wong, Ting-Kam Leonard and Chen, Ricky T. Q. and Duvenaud, David K.},
    booktitle = {Proceedings of The 2nd Symposium on Advances in Approximate Bayesian Inference},
    pages     = {1--28},
    year      = {2020},
    volume    = {118},
    series    = {Proceedings of Machine Learning Research},
    publisher = {PMLR}
}
```

For the dataset preprocessing and ODE²VAE model ([paper link](https://arxiv.org/abs/1905.10994)):
```bibtex
@inproceedings{yildiz2019ode2vae,
    title     = {ODE2VAE: Deep generative second order ODEs with Bayesian neural networks},
    author    = {Yildiz, Cagatay and Heinonen, Markus and Lahdesmaki, Harri},
    booktitle = {Advances in Neural Information Processing Systems},
    volume    = {32},
    year      = {2019},
    publisher = {Curran Associates, Inc.}
}
```