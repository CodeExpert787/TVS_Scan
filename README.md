# PCOS Meal Plan Generator

A web-based application that generates personalized meal plans for PCOS (Polycystic Ovary Syndrome) patients using machine learning. The application analyzes patient data and provides customized meal plans, exercise recommendations, and supplement suggestions.

## Features

- Upload and process PCOS patient data
- Train machine learning models on patient data
- Generate personalized meal plans based on:
  - Patient's physical data (weight, height, BMI)
  - PCOS type
  - Age and other factors
- Real-time training status monitoring
- Web-based user interface
- Detailed meal plans including:
  - Breakfast, lunch, and dinner options
  - Exercise recommendations
  - Supplement suggestions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pcos-meal-plan-generator.git
cd pcos-meal-plan-generator
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the folder structure:
```bash
python setup_folders.py
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Upload training data:
   - Click "Choose File" in the Model Training section
   - Select a DOCX file containing PCOS patient data
   - Click "Upload File"

4. Start training:
   - Click "Start Training"
   - Monitor the training progress

5. Generate meal plans:
   - Fill in the patient information
   - Select PCOS types
   - Click "Generate Meal Plan"

## Project Structure

```
pcos-meal-plan-generator/
├── data/
│   ├── raw/           # Raw training data
│   └── processed/     # Processed data
├── models/            # Trained models
├── logs/             # Application logs
├── static/           # Static web files
│   ├── css/         # CSS files
│   └── js/          # JavaScript files
├── templates/        # HTML templates
├── main.py          # FastAPI application
├── setup_folders.py # Folder structure setup
├── requirements.txt # Project dependencies
└── README.md        # Project documentation
```

## API Endpoints

- `GET /`: Web interface
- `POST /upload-training-data`: Upload training data
- `POST /start-training`: Start model training
- `GET /training-status`: Check training status
- `POST /generate-meal-plan`: Generate meal plan
- `GET /pcos-types`: Get available PCOS types

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 