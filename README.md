# sota-data-augmentation-and-optimizers
This repository contains some of the latest data augmentation techniques and optimizers for image classification using pytorch and the CIFAR10 dataset

## Data Augmentation Technique
This repository implements the following data augmentation techniques. The links to the pappers and and pytorch code references are associated accordingly (some with slight modification).

CutOut ([Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf, [code](https://github.com/uoguelph-mlrg/Cutout))

AutoAugment ([AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501v1), [code](https://github.com/DeepVoltaire/AutoAugment))

RandAugment ([RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/pdf/1909.13719v2.pdf), [code](https://github.com/ildoonet/pytorch-randaugment))

AugMix ([AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/pdf/1912.02781v1.pdf), [code](https://github.com/google-research/augmix))

ISDA ([Implicit Semantic Data Augmentation for Deep Networks](https://arxiv.org/pdf/1909.12220v4.pdf), [code](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks))

## Architecture
Custom convolutional neural network model that makes use of depthwise convolution and [squeeze-and-excitation](https://arxiv.org/pdf/1709.01507v4.pdf) and the [mish](https://arxiv.org/pdf/1908.08681v2.pdf) activation function.

You may use any model of choice, but slight modification is needed in order to implement ISDA.

## Optimizers
AdaMod

DeepMemory ([code](https://github.com/lessw2020/Best-Deep-Learning-Optimizers)) 

Ranger ([RAdam](https://arxiv.org/pdf/1908.03265v1.pdf) with [LookAhead](https://arxiv.org/abs/1907.08610)) 

AdaLook (AdaMod with LookAhead (mine)) 

## Loss function
I made use of cross entropy loss with label smoothing as implemented [here](https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py)

## Installation
[Install Python 3 with anaconda](https://www.continuum.io/downloads)

    $ git clone https://github.com/etetteh/sota-data-augmentation-and-optimizers
    $ cd sota-data-augmentation-and-optimizers
    $ pip install -r requirement.txt

## Implementation
### Sample Notebook
The [sample_notenook.ipynb] contains code to play with the various augmentaion and optimizer techniques. Simply comment or uncomment appropriate lines.


### Scripts
You may run the following line of code in your bash terminal


    $ python main.py -h
    usage: main.py [-h] [--cutout] [--autoaug] [--randaug] [--augmix] [--adamod]
               [--adalook] [--deepmemory] [--ranger] [--resume] [--path PATH]
               [--epochs EPOCHS] [--num_workers NUM_WORKERS]
               [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
               [--weight_decay WEIGHT_DECAY] [--print_freq PRINT_FREQ]
               [--seed SEED]

    Data Augmentation Techniques on CIFAR10 with PyTorch.

    optional arguments:
    -h, --help            show this help message and exit
    --cutout              Using CutOut data augmentation technique.
    --autoaug             Using AutoAugment data augmentation technique.
    --randaug             Using RandAugment data augmentation technique.
    --augmix              Using AugMixt data augmentation technique.
    --adamod              Use AdaMod optimizer
    --adalook             Use AdaMod+LookAhead optimizer
    --deepmemory          Use DeepMemory optimizer
    --ranger              Use RAdam+LookAhead optimizer
    --resume, -r          resume training from checkpoint.
    --path PATH           path to checkpoint. pass augmentation name
    --epochs EPOCHS, -e EPOCHS
                        Number of training epochs.
    --num_workers NUM_WORKERS
                        Number of CPUs.
    --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        input batch size for training.
    --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        learning rate.
    --weight_decay WEIGHT_DECAY, -wd WEIGHT_DECAY
                        weight decay.
    --print_freq PRINT_FREQ, -pf PRINT_FREQ
                        Number of iterations to print out results
    --seed SEED           random seed

Example: 
To train using adamod optimizer and augmix augmentation for 100 epochs, run:

    $ python main.py --adamod --augmix --epochs 100

To resume training, run:

    $ python main.py --resume --adamod --augmix --epochs 100
    
To train using isda, adamod optimizer and augmix augmentation for 100 epochs, run:

    $ python main_isda.py --adamod --augmix --epochs 100

## Coming Soon...
Results with my custom model and other models.

## To do
CutMix ([CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899v2.pdf), [code](https://github.com/clovaai/CutMix-PyTorch))

AdvProp ([Adversarial Examples Improve Image Recognition](https://arxiv.org/pdf/1911.09665v1.pdf))

Memory Efficient Mish
