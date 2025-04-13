# 🥔 Potato Leaf Disease Classification

A deep learning-based web application for classifying potato leaf diseases into three categories: Late Blight, Healthy, and Early Blight.

## 📋 Project Overview

This project uses a Convolutional Neural Network (CNN) to classify potato leaf images into three categories:
- Late Blight
- Healthy
- Early Blight

The model was trained on a comprehensive dataset of potato leaf images and achieves high accuracy in disease classification.

## 🛠️ Technical Stack

- **Deep Learning Framework**: TensorFlow 2.10.0
- **Web Interface**: Streamlit
- **Image Processing**: Pillow, OpenCV
- **Data Processing**: NumPy
- **Visualization**: Matplotlib

## 📁 Project Structure

```
potato-disease-classification/
├── app.py                  # Streamlit web application
├── potato_disease_model1.keras  # Trained model file
├── requirements.txt        # Python dependencies
└── Potato Leaf Disease Classification project 2.ipynb  # Model development notebook
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd potato-disease-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv potato_env
.\potato_env\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Activate the virtual environment:
```bash
.\potato_env\Scripts\activate  # On Windows
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to:
```
http://localhost:8501
```

## 🖼️ Using the Application

1. **Upload an Image**:
   - Click the "Choose a potato leaf image..." button
   - Select an image of a potato leaf (supports JPG, JPEG, PNG formats)

2. **View Results**:
   - The application will display:
     - Uploaded image
     - Predicted disease class
     - Confidence score
     - Probability distribution across all classes
     - Health status message

## 🧠 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 256x256 pixels
- **Output Classes**: 3 (Late Blight, Healthy, Early Blight)
- **Training Data**: Comprehensive dataset of potato leaf images
- **Performance**: High accuracy in disease classification

## 📊 Class Descriptions

1. **Late Blight**
   - Caused by the fungus-like organism *Phytophthora infestans*
   - Symptoms: Dark, water-soaked spots, white fuzzy growth
   - Can destroy crops within days

2. **Healthy**
   - Normal potato leaves without disease
   - Green, uniform appearance
   - No spots or discoloration

3. **Early Blight**
   - Caused by the fungus *Alternaria solani*
   - Symptoms: Dark brown spots with concentric rings
   - Typically appears earlier in the growing season

## 🛠️ Development

The model was developed in the Jupyter notebook `Potato Leaf Disease Classification project 2.ipynb`, which contains:
- Data preprocessing
- Model architecture
- Training process
- Evaluation metrics
- Model saving

## 📝 Requirements

Listed in `requirements.txt`:
```
tensorflow>=2.10.0
streamlit>=1.22.0
pillow>=9.5.0
numpy>=1.23.5
matplotlib>=3.5.0
opencv-python>=4.5.5
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset providers
- Open-source community
- Agricultural experts for validation 
