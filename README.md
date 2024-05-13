# Transformer-based-CNN-Approach-for-RS-Scene-Classification

Feature extraction in remote sensing is a challenging yet crucial operation for scene classification because of cloud cover and overlapping edges present in the data. We present an analysis of different deep learning architectures on multiple scene classification datasets to understand the features and weigh the advantages of one or more functional blocks (such as convolution and attention mechanism) connected in different convolutional neural networks. The work considers five open-source benchmark datasets: UC-Merced Land Use, WHU-RS19, Optimal-31, RSI-CB256, and MLRSNet, that have been openly made available to the research community. Architectures such as VGG-16, Resnet50, EfficientNetB3, Vision Transformers (ViT), Swin Transformers, and ConvNeXt are used to perform this task. Though the comparison between deep learning models for scene classification has been made, the comparison between different Transformer-based architectures and convolution-based architectures has not been systematically addressed in the remote sensing literature. We have obtained a new benchmark that exceeds the state-of-the-art results for all the datasets on a 90:10 train-test split.

Keywords: Convolutional Neural Networks; Remote Sensing; Scene Classification; fastai

### Codes: - Refer to the TrainerNotebookActual.ipynb notebook for training from scratch
###        - If you have the pre-trained weights and wish to make predictions, refer to TrainerNotebookPredictions.ipynb
### Links to download the open-source datasets:
###    1) UC Merced Land Use    : http://weegee.vision.ucmerced.edu/datasets/landuse.html
###    2) WHU-RS19              : https://captain-whu.github.io/BED4RS/
###    3) Optimal-31            : https://drive.google.com/file/d/1Fk9a0DW8UyyQsR8dP2Qdakmr69NVBhq9/view
###    4) RSI-CB256             : https://github.com/lehaifeng/RSI-CB
###    5) MLRSNet               : https://data.mendeley.com/datasets/7j9bv9vwsx/2
### {PreTrained Weights of models can be made available on request - Mail to arrun.sivasubramanian@gmail.com} 
### Please star the repository if you like this work :)
Experimental Setup: 

The pre-trained weights for transfer learning were acquired for PyTorch from the timm module. Finding the optimal learning rate for faster convergence was done using the lr_finder method. Since our work considers a fixed batch size of 8, this method finds the best learning rate by splitting the data into batches and assigning different learning rates to each batch. Then, the learning rate of the batch with a small loss is considered the optimal learning rate. The results were computed using 4 Nvidia RTX5000 GPUs, and the fastai distributed framework helps in parallel computation. The setup has a CPU RAM of 128 GB, GPU RAM of 64 GB, and a 28-core processor. For comparison, we use accuracy (the ratio between correctly classified instances to the total instances present) and the F1 score (the harmonic226 mean between precision and recall) as the evaluation metric since we need to minimize the false positive and the false pessimistic predictions in the confusion matrix. For saving the models, we prioritize minimizing the validation loss. 

Proposed Methodology Block Diagram:

![Figure14_Methodology](https://github.com/argon125/Transformer-based-CNN-Approach-for-RS-Scene-Classification/assets/64146402/7e24f4f0-c7fa-4f9e-afe1-62b85d8b0de1)
