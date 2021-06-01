# Merantix-Labs: DAAIN

This is the code for our paper **DAAIN: Detection of Anomalous and Adversarial Input using Normalizing Flows**.

## Assumptions

There are assumptions:

- The training data PerturbedDataset makes some assumptions about the data:
    - the `ignore_index` is 255
    - `num_classes` = 19
    - the images are resized with `size == 512`

## Module Overview

A selection of the files with some pointers what to find where

```
├── configs                                   # The yaml configs
│   ├── activation_spaces
│   │   └── esp_net_256_512.yaml
│   ├── backbone
│   │   ├── esp_dropout.yaml
│   │   └── esp_net.yaml
│   ├── dataset_paths
│   │   ├── bdd100k.yaml
│   │   └── cityscapes.yaml
│   ├── data_creation.yaml                    # Used to create the training and testing data in one go
│   ├── detection_inference.yaml              # Used for inference
│   ├── detection_training.yaml               # Used for training
│   ├── esp_dropout_training.yaml             # Used to train the MC dropout baseline
│   └── paths.yaml
├── README.md                                 # This file!
├── requirements.in                           # The requirements
├── setup.py
└── src
   └── daain
       ├── backbones                          # Definitions of the backbones, currently only a slighlty modified version
       │   │                                  # of the ESPNet was tested
       │   ├── esp_dropout_net
       │   │   ├── esp_dropout_net.py
       │   │   ├── __init__.py
       │   │   ├── lightning_module.py
       │   │   └── trainer
       │   │       ├── criteria.py
       │   │       ├── data.py
       │   │       ├── dataset_collate.py
       │   │       ├── data_statistics.py
       │   │       ├── __init__.py
       │   │       ├── iou_eval.py
       │   │       ├── README.md
       │   │       ├── trainer.py            # launch this file to train the ESPDropoutNet
       │   │       ├── transformations.py
       │   │       └── visualize_graph.py
       │   └── esp_net
       │       ├── espnet.py                 # Definition of the CustomESPNet
       │       └── layers.py
       ├── baseline
       │   ├── maximum_softmax_probability.py
       │   ├── max_logit.py
       │   └── monte_carlo_dropout.py
       ├── config_schema
       ├── constants.py                      # Some constants, the last thing to refactor...
       ├── data                              # General data classes
       │   ├── datasets
       │   │   ├── bdd100k_dataset.py
       │   │   ├── cityscapes_dataset.py
       │   │   ├── labels
       │   │   │   ├── bdd100k.py
       │   │   │   ├── cityscape.py
       │   │   └── semantic_segmentation_dataset.py
       │   ├── activations_dataset.py        # This class loads the recorded activations
       │   └── perturbed_dataset.py          # This class loads the attacked images
       ├── model
       │   ├── aggregation_mode.py           # Not interesting for inference
       │   ├── classifiers.py                # All classifiers used are defined here
       │   ├── model.py                      # Probably the most important module. Check this for an example on how
       │   │                                 # to used the detection model and how to load the parts
       │   │                                 # (normalising_flow & classifier)
       │   └── normalising_flow
       │       ├── coupling_blocks
       │       │   ├── attention_blocks
       │       │   ├── causal_coupling_bock.py  # WIP
       │       │   └── subnet_constructors.py
       │       └── lightning_module.py
       ├── scripts
       │   └── data_creation.py              # Use this file to create the training and testing data
       ├── trainer                           # Trainer of the full detection model
       │   ├── data.py                       # Loading the data...
       │   └── trainer.py
       ├── utils                             # General utils
       └── visualisations                    # Visualisation helpers
```

## Parts

In general the model consists of two parts:

- Normalising FLow
- Classifier / Scoring method

Both have to be trained separately, depending on the classifier. Some are parameter free (except for the threshold).

The general idea can be summarised:

1. Record the activations of the backbone model at specific locations during a forward pass.
2. Transform the recorded activations using a normalising flow and map them to a standard Gaussian for each variable.
3. Apply some simple (mostly distance based) classifier on the transformed activations to get the anomaly score.

## Training & Inference Process

1. Generate perturbed and adversarial images. We do not provide code for this step.
2. Generate the activations using `src/daain/scripts/data_creation.py`
3. Train the detection model using `src/daain/trainer/trainer.py`
4. Use `src/daain/model/model.py` to load the trained model and use it to get the anomaly score (the probability that the
   input was anomalous).

