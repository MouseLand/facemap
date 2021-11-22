"""
Facemap (UNet) models trained for generating pose estimates. Contains functions for: 
- downloading pre-trained models
- Model class 
"""

# Load pytorch model from web
def load_pytorch_model(url):
    