# EE641 Project: How Medical Image Tell? 
In this project, we proposed a deep learning architecture - **AIMR-MediTell** that can detect and highlight the malfunctions of patients based on the patient’s chest X-ray screenings and generate understandable and accurate diagnostic reports. Our model architecture is composed by the Instance Segmentation and Image-based Text Generation. Ours model architecture is shown in Figure below. 

<img src="https://user-images.githubusercontent.com/63425702/236669671-2c4f58f4-8785-48d7-a5a3-012b024a08d4.jpg" width="700" height="350">

## Expected Model Functionality
Our Model takes a clinical frontal view Chest X-ray as the input image. The Instance Segmentation model generates the masked image and the Attention model can produce the diagnosis report from the masked image. The generated report could support experts in medical field to interpret the meaning of Chest X-ray. 

<img src="https://user-images.githubusercontent.com/63425702/236601842-a55dd16c-da53-4b19-90d6-a25c82c5dece.png" width="700" height="350">

## File Structure 

    .
    ├── image_segmentation_scripts    # Codes for training instance segmentation models and inferrence scripts
        ├── Final_Version_Instance_Segmentation_with_a_Novel_Mask_RCNN_14_Classes_Without_Outputs.ipynb
        ├── Final_Version_Instance_Segmentation_with_a_Novel_Mask_RCNN_14_Classes_Without_Outputs.py
    ├── performance_log               # Includes evaluation and performance stats, e.g, loss curve, accuracy curve
    ├── sample_data                   # Sample data of models' inputs, outputs, e.g, masked RGB images
    ├── Data_Loading.py               # Data Loading files for text generation model
    ├── Image_Processing.py           # Process the original datasets and generate train, validation and test datasets for text generation training
    ├── Text_Generate_Model.py        # Text generation model architecture: Encoder-Deconder with Attention
    ├── Training.py                   # Training and test scripts for text generation
    ├── Predict_Findings.py           # Predict the text finding based on given images on the trained text generation model
    ├── utils.py                      # Helper functions such as plotting loss curve, saving checkpoints
    ├── Trained_Model_Information.txt # Accessing the Trained Models, Ways of Loading and File Sizes
    ├── requirements.txt              # environments and required python packages
    └── README.md


## Tutorial
### Text Generation

+ Process the original dataset and split the train, validation and test datasets and save them. \
Run `python3 Image_Processing.py`
```Python
Image_Processing.py
# Change the the path_names in the main() function 
```


+ Training: Train the text generator on the training datast and evaluate on validation dataset during the training process. The final model will be finally evaluated on the test dataset. \
 Run `python3 Training.py`
```Python
Training.py
# change the file paths in the main() function
# Change the model hyperparameters inb the train() function
# The final model will be saved in the saved_model_path
```

+ Prediction: Predict the findings based on given images on the trained_model after training., and plot the corresponding attention map. \
     Run `python3 Predict_Findings.py`
```Python
Predict_Findings.py
# Change the image path in the main() function
# Set plot_att_map = True to plot attention map
```

### Image Segmentation

+ Training/Regeneration of Results and Synthesized Masked Images:
     Run `python3 Final_Version_Instance_Segmentation_with_a_Novel_Mask_RCNN_14_Classes_Without_Outputs.py`
```Python

Final_Version_Instance_Segmentation_with_a_Novel_Mask_RCNN_14_Classes_Without_Outputs.py

# Download and install the libraries through the requirements.txt file.

# Change the directory in the line below to your “saved_directory” where the zipped ChestXDet Dataset 
resides: “get_ipython().system('unzip saved_directory/ChestXDet_Dataset.zip’)”

# Make sure that you have the metadata files, i.e. ‘ChestX_Det_train.json’ and ‘ChestX_Det_test.json’ 
in your “saved_directory”

# Change the directory in the line below to your “saved_directory” where the zipped Indiana University Dataset
resides as “images_normalized.zip”:
“get_ipython().system('unzip saved_directory/images_normalized.zip’)”

# Make sure that you have the metadata files, i.e. ‘indiana_images_info.csv’ in your “saved_directory”

# Make sure the change the directories through which the files are read either via “json.open” or 
“pandas.read_csv” to “saved_directory/name_of_the_file_read”.


# Mount and Allow access to your Google Drive and create the following directories through your drive:
        '/content/drive/MyDrive/ee641/Project_Datasets/Model_Checkpoints_1’
        '/content/drive/MyDrive/ee641/Project_Datasets/Model_Checkpoints_2’
        '/content/drive/MyDrive/ee641/Project_Datasets/Model_Checkpoints_3’
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Images_Instance_Segmentation_1/’
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Images_Instance_Segmentation_2/‘
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Images_Instance_Segmentation_3/’
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Masked_Synthesized_Indiana_University_Dataset/’
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Non_Masked_Synthesized_Indiana_University_Dataset/’
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_05_Threshold/’
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_06_Threshold/’
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_07_Threshold/’
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_08_Threshold/’
        ‘/content/drive/MyDrive/ee641/Project_Datasets/Masking_Strategies_Best_Model_Otsu_Threshold/’

# After that feel free to reinitialize the training process after which you can regenerate the results for. 
Please kindly keep in mind that you need a GPU RAM Space of at least 40 GB as a requirement to go through 
the time-consuming training process. Otherwise, you need to change certain parameters such as the batch size 
during the training or the image sizes being fed to the models manually to be inline with your computational 
requirements. Also, please kindly refer to the "Trained_Model_Information.txt" on ways to access the pretrained model files.
```
