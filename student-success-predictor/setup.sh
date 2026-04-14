#!/bin/bash
echo "🎓 Setting up Student Success Predictor..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data
python src/data/load_data.py

# Train models
python src/models/train.py

echo "✅ Setup complete!"
echo "Run: streamlit run app/streamlit_app.py"
