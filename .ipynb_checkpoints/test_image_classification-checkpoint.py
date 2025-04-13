import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Define sample images for each category
sample_images = {
    "Late Blight": "https://www.agriculture.com.ph/wp-content/uploads/2019/10/late-blight.jpg",
    "Healthy": "https://media.istockphoto.com/id/1400057561/photo/potato-plant.jpg?s=612x612&w=0&k=20&c=Pyz5CxpqOlCAkTYV8f3XBT_xEwXbBq0xaCWJR_Gzs-M=",
    "Early Blight": "https://www.planetnatural.com/wp-content/uploads/2012/12/potato-early-blight.jpg"
}

# Load model
print("Loading model...")
model = tf.keras.models.load_model('potato_disease_model.keras')

# Function to preprocess image
def preprocess_image(image):
    # Resize image to 256x256
    image = image.resize((256, 256))
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define possible class orders to test
possible_orders = [
    ['Late Blight', 'Healthy', 'Early Blight'],  # Current order in app.py
    ['Early Blight', 'Late Blight', 'Healthy'],  # Potential order 1
    ['Healthy', 'Late Blight', 'Early Blight'],  # Potential order 2
    ['Early Blight', 'Healthy', 'Late Blight'],  # Potential order 3
    ['Healthy', 'Early Blight', 'Late Blight'],  # Potential order 4
    ['Late Blight', 'Early Blight', 'Healthy']   # Potential order 5
]

# Download and test each sample image
for true_label, url in sample_images.items():
    print(f"\n\n==== Testing {true_label} Image ====")
    
    # Download image
    print(f"Downloading {true_label} image from {url}")
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        print(f"Image loaded successfully")
        
        # Display image
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.title(f"Sample {true_label} Image")
        plt.axis('off')
        plt.savefig(f"sample_{true_label.replace(' ', '_')}.png")
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        
        print(f"Raw prediction: {prediction[0]}")
        
        # Get top prediction index and value
        top_index = np.argmax(prediction[0])
        top_value = prediction[0][top_index]
        
        print(f"Model predicts index {top_index} with confidence {top_value:.4f}")
        
        # Check each possible class order
        print("\nTesting different class orderings:")
        
        for i, class_order in enumerate(possible_orders):
            predicted_class = class_order[top_index]
            match = predicted_class == true_label
            print(f"Order {i+1} ({' -> '.join(class_order)}): Predicts {predicted_class}")
            print(f"  {'✅ CORRECT' if match else '❌ INCORRECT'}")
    
    except Exception as e:
        print(f"Error processing {true_label} image: {e}")

print("\nDone!") 