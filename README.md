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

### Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MaksymKashuba/airbus-ship-detection-challenge-kaggle.git
   cd airbus-ship-detection-challenge-kaggle
   ```

2. **Set Up the Environment**:
   Create a virtual environment and install the required dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare the Data**:
   - Download the dataset from the Kaggle competition page.
   - Extract the dataset into the `data/` directory.

4. **Train the Model**:
   Run the training script:
   ```bash
   python train.py
   ```

5. **Generate Predictions**:
   Use the inference script to make predictions:
   ```bash
   python inference.py
   ```

6. **Analyze Results**:
   Open the Jupyter Notebook interface to analyze the results. Use the provided notebooks (`analysis.ipynb` and `test.ipynb`) for data analysis and model evaluation.
