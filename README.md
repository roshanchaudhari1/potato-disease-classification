🥔 Potato Leaf Disease Classification
A deep learning-powered web application that classifies potato leaf images into three categories: Late Blight, Healthy, and Early Blight. Built using a CNN model and deployed with Streamlit.

📁 Project Structure
bash
Copy
Edit
potato-disease-classification/
├── .ipynb_checkpoints/                     # Jupyter auto-save files (can be ignored)
├── PlantVillage/                           # Raw dataset folder (from PlantVillage)
├── datset/                                 # Possibly preprocessed or organized dataset
├── Potato_Unknown leafs/                   # Images with unknown classification
├── Potato Leaf Disease Classification project 2.ipynb  # Jupyter notebook for model training
├── README.md                               # Project overview and instructions
├── app.py                                  # Streamlit web app
├── git/                                    # (Unclear content – maybe backup or config folder)
├── potato_disease_model1.keras             # Trained CNN model
├── requirements.txt                        # Python dependencies
📋 Project Overview
This project uses a Convolutional Neural Network (CNN) to identify and classify potato leaf diseases. It features a user-friendly Streamlit interface where users can upload an image and receive:

Disease prediction

Confidence scores

Health tips based on class

🛠️ Technologies Used
TensorFlow 2.10.0 – Model training and inference

Streamlit – Interactive web interface

Pillow, OpenCV – Image processing

NumPy – Numerical operations

Matplotlib – Visualizations in the notebook

🚀 Getting Started
Prerequisites
Python 3.10 or later

pip package manager

Installation
Clone the Repository

bash
Copy
Edit
git clone https://github.com/roshanchaudhari1/potato-disease-classification.git
cd potato-disease-classification
Create a Virtual Environment

bash
Copy
Edit
python -m venv potato_env
.\potato_env\Scripts\activate  # For Windows
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
▶️ Running the Application
Activate the environment:

bash
Copy
Edit
.\potato_env\Scripts\activate
Start the Streamlit server:

bash
Copy
Edit
streamlit run app.py
Open your browser and go to:

arduino
Copy
Edit
http://localhost:8501
🖼️ How to Use the App
Upload an Image of a potato leaf (JPG, JPEG, PNG)

The app will:

Display the uploaded image

Predict the disease

Show confidence scores

Display advice or status based on prediction

🧠 Model Details
Model Architecture: CNN with convolution, pooling, dense layers

Image Size: Resized to 256x256

Classes: Late Blight, Healthy, Early Blight

Model File: potato_disease_model1.keras

📊 Class Breakdown
Class	Description
Late Blight	Caused by Phytophthora infestans. Rapid spread. Water-soaked lesions.
Healthy	No disease symptoms. Uniform green.
Early Blight	Caused by Alternaria solani. Brown concentric rings on older leaves.
📓 Notebook
The notebook Potato Leaf Disease Classification project 2.ipynb includes:

Data loading from PlantVillage/ and datset/

Image preprocessing

Model training

Evaluation metrics

Saving .keras model

📝 Dependencies (requirements.txt)
shell
Copy
Edit
tensorflow>=2.10.0
streamlit>=1.22.0
pillow>=9.5.0
numpy>=1.23.5
matplotlib>=3.5.0
opencv-python>=4.5.5
🤝 Contributing
Found a bug or want to enhance the model/UI?
Pull requests and suggestions are welcome!

📄 License
This project is licensed under the MIT License.

🙏 Acknowledgments
PlantVillage Dataset

TensorFlow and Streamlit contributors

Agricultural experts and open-source community

