# EE641 Project: How Medical Image Tell? 
In this project, we proposed a deep learning architecture - **AIMR-MediTell** that can detect and highlight the malfunctions of patients based on the patient’s chest X-ray screenings and generate understandable and accurate diagnostic reports. Our model architecture is composed by the Instance Segmentation and Image-based Text Generation. Ours model architecture is shown in Figure below. 

<img src="https://user-images.githubusercontent.com/63425702/236669671-2c4f58f4-8785-48d7-a5a3-012b024a08d4.jpg" width="700" height="350">

## Expected Model Functionality
Our Model takes a clinical frontal view Chest X-ray as the input image. The Instance Segmentation model generates the masked image and the Attention model can produce the diagnosis report from the masked image. The generated report could support experts in medical field to interpret the meaning of Chest X-ray. 

<img src="https://user-images.githubusercontent.com/63425702/236601842-a55dd16c-da53-4b19-90d6-a25c82c5dece.png" width="700" height="350">

## File Structure 

    .
    ├── image_segmentation            # Codes for training instance segmentation models and inferrence scripts
        ├── Final_Version_Instance_Segmentation_with_a_Novel_Mask_RCNN_14_Classes_Without_Outputs.ipynb
        ├── Final_Version_Instance_Segmentation_with_a_Novel_Mask_RCNN_14_Classes_Without_Outputs.py
    ├── Data_Loading.py               # Data Loading file
    ├── Image_Processing.py           # Process the original datasets and generate train, validation and test datasets
    ├── Text_Generate_Model.py        # Text generation model architecture: Encoder-Deconder with Attention
    ├── Training.py                   # Training and test scripts
    ├── Prediction.py                 # Predict the text finding based on given images on the trained model
    ├── utils.py                      # Helper functions such as plotting loss curve, saving checkpoints
    ├── Trained_Model_Information.txt # Accessing the Trained Models, Ways of Loading and File Sizes
    ├── requirements.txt              # environments and required python packages
    └── README.md


## Tutorial
### Image Segmentaion

+ Get masked image from original dataset

    Run the ``

+ The 

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
     Run `python3 Prediction.py`
```Python
Prediction.py
# Change the image path in the main() function
# Set plot_att_map = True to plot attention map
```
