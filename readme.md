## Airbus Ship Detection Challenge

### Project Overview
This project is about finding ships in satellite images using machine learning and deep learning. It focuses on image processing, finding objects, and segmenting images with neural networks.

### Components
- **Python Files**: 
  - `inference.py`: Generates predictions.
  - `train.py`: Trains a deep learning model using U-Net, with data augmentation and upsampling.
  - `utils.py`: Has tools for decoding image masks and custom loss metrics for image segmentation.
  
- **Jupyter Notebooks**: 
  - `analysis.ipynb`: Analyzes data and evaluates the model.
  - `test.ipynb`: Tests the model with TensorFlow and OpenCV.

### Dependencies
- **Deep Learning**: TensorFlow, Keras
- **Data Handling**: NumPy, Pandas
- **Image Processing**: OpenCV, Scikit-Image
- **Visualization**: Matplotlib
