# CENSER
This repository is implementation of [CENSER paper](https://bit.ly/2QWcLwk).

## Environment Setup
All the dependencies can be installed using

    pip install -r requirements.txt
    
## Training
* Download the [pretrained model](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) trained on VGGFace to directory named 'pretrained'.
* Copy a kinship dataset to a directory named 'dataset', with images corresponding to each class in a sub-directory. 
* For training navigate to 'src' directory and run

      python train_tripletloss.py --logs_base_dir ../trained/logs --models_base_dir ../trained/trained_model --data_dir ../dataset --image_size 160 --pretrained_model ../pretrained_model/20180402-114759 --model_def models.inception_resnet_v1 --optimizer RMSPROP --learning_rate 0.01 --weight_decay 1e-4 --max_nrof_epochs 10 --embedding_size 512 --images_per_person 2 --gpu_memory_fraction 1
      
* Model can be tested using 

      python test.py
      
## Age-Gender detection
* The code for age-gender detection mentioned in the paper is made available [here](https://github.com/RaghuHemadri/Age_Gender_Estimator). 
* The code is made available seperately for high customizability.
