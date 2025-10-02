# Image Upload and Prediction Web Application

This project is a simple web application built using Flask that allows users to upload images and receive predictions based on a key validation. The application is designed to be visually appealing, simple, responsive, and user-friendly.

## Project Structure

```
web
├── app.py                # Main Flask application
├── static
│   ├── css
│   │   └── style.css     # CSS styles for the web application
│   └── js
│       └── script.js     # JavaScript for handling uploads and displaying results
├── templates
│   └── index.html        # HTML file for the user interface
├── uploads               # Directory for storing uploaded images
├── output                # Directory for storing prediction results
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the Repository**: 
   Clone this repository to your local machine.

2. **Install Requirements**: 
   Navigate to the project directory and install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**: 
   Start the Flask application by running:
   ```
   python app.py
   ```
   The application will run on `http://127.0.0.1:12345`.

4. **Access the Web Interface**: 
   Open your web browser and go to `http://127.0.0.1:12345` to access the image upload interface.

## Usage Guidelines

- Use the provided form to upload an image and enter the required key for validation.
- After submission, the application will process the image and display the prediction results on the same page.
- Uploaded images will be stored in the `uploads/` directory, and the prediction results will be saved in the `output/` directory.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. Your feedback and suggestions are welcome!