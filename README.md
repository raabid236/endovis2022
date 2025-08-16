# endovis2022
Semantic segmentation aspect of the SAR-RARP50 challenge

The code was built using python 3.11 and uses Deeplabv3_resnet50 model from torchvision achieving 0.73 and 0.71 miou.

Training script is available as a python notebook that was used in google collab. To run the code, adjust the section starting/labelled as 'Arguments'. The parameters used for training are pre-filled in the scripts: 
- endovis2022_focal.ipynb trains using CE and focal combined loss
- endovis2022_pytorch_nofocal.ipynb trains using CE loss

The following arguments are needed:
- train_dir: folder containing training images
- val_dir: folder containing validation images
- out_dir: folder where trained model will be stored
- epochs: number of epochs to run
- batch_size: batch size for training
- img_size: image size for training. For computational limitations, this was set to 256
- lr: learning rate
- num_workers: number of cores to be used
- num_classes: number of classes/labels. For this dataset, this should be 10
- device='cuda' if torch.cuda.is_available() else 'cpu'

Two pretrained models are available in the models folder.
- models/ce/best_model.pth was trained using CE loss for 16 epochs
- models/cefocal/best_model.pth was trained using CE and focal loss for 7 epochs

Inference can be undertaken by executing the inference_pytorch_semantic2.py script. The script load the model and datset and saves both teh prediction maps and the overlayed masks on original image frame and computes the miou metric for the entire dataset. The following arguments are needed:
- output_dir: folder were predictions will be stored in similar format to input
- input_dir: folder containing test images
- model_ckpt: path to the trained model

To execute the entire workflow (adjust arguments in the code as needed):
- download SAR-RARP50 dataset
- install required libraries using preferred package manager
- run data_extract.py 
- run resized.py
- run endovis2022_focal.ipynb or endovis2022_pytorch_nofocal.ipynb as preferred for training
- run inference_pytorch_semantic2.py for inference