# Oral Cancer Screening System

An AI-powered oral cancer screening system that uses deep learning and Grad-CAM visualization to detect potential cancer in oral cavity images. The system provides an interactive interface for collecting patient demographics, analyzing images, and generating comprehensive PDF reports.

## Features

- 🔬 **Deep Learning Model**: ResNet50-based binary classifier for cancer/non-cancer prediction
- 🔥 **Grad-CAM Visualization**: Visual explanation of model predictions using Gradient-weighted Class Activation Mapping
- 📊 **Interactive Interface**: User-friendly widgets for data collection and image upload
- 📄 **PDF Report Generation**: Automated generation of detailed diagnostic reports
- 👤 **Demographics Collection**: Comprehensive patient information including age, gender, and risk factors

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Jupyter Notebook support
- VS Code with Jupyter extension (recommended) or Jupyter Lab/Notebook

## Installation

### Step 1: Clone or Download the Project

Download the project files to your local machine and navigate to the project directory:

```bash
cd path/to/Delivery
```

### Step 2: Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies

Install all required packages using the requirements file:

```bash
pip install -r requirements.txt
```

This will install the following packages:
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities for PyTorch
- `numpy` - Numerical computing library
- `Pillow` - Image processing library
- `opencv-python` - Computer vision library
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning utilities
- `ipywidgets` - Interactive widgets for Jupyter
- `reportlab` - PDF generation library
- `google-genai` - Google Generative AI (for report enhancement)
- `python-dotenv` - Environment variable management
- `tensorflow` - Deep learning framework (supporting libraries)
- `tqdm` - Progress bar utilities

### Step 4: Set Up Environment Variables

Create a `.env` file in the project directory to store your Gemini API key:

1. Create a new file named `.env` in the `Delivery` folder (same directory as `main.ipynb`)

2. Add the following line to the `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Getting your Gemini API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the generated API key and replace `your_api_key_here` in your `.env` file

**Important**: Never commit the `.env` file to version control. It contains sensitive API credentials.

### Step 5: Verify Model File

Ensure the trained model file `best_model_224_resnet50.pt` is present in the project directory.

## Project Structure

```
Delivery/
│
├── main.ipynb                      # Main Jupyter notebook (run this)
├── report.py                       # PDF report generation module
├── requirements.txt                # Python dependencies
├── best_model_224_resnet50.pt     # Pre-trained model weights
├── README.md                       # This file
├── __pycache__/                   # Python cache directory
└── Eval/                          # Evaluation resources (if any)
```

## Usage

### Running the Application

#### Option 1: Using VS Code (Recommended)

1. **Open VS Code** and navigate to the project folder
2. **Open `main.ipynb`** in VS Code
3. **Select Python Kernel**: 
   - Click on "Select Kernel" in the top-right corner
   - Choose your Python environment (preferably the venv you created)
4. **Run the First Cell**: Click the ▶️ (Play) button next to the first code cell or press `Shift + Enter`
5. The interactive interface will appear below the cell

#### Option 2: Using Jupyter Notebook/Lab

1. **Launch Jupyter**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```
2. **Open `main.ipynb`** from the Jupyter interface
3. **Run the first cell** by clicking the Run button or pressing `Shift + Enter`

### Using the System

Once you run the notebook cell, follow these steps:

#### Step 1: Enter Patient Demographics

1. Fill in the following information:
   - **Age**: Enter patient's age (numeric value)
   - **Gender**: Select from dropdown (Male/Female/Other/Prefer not to say)
   - **Smoking Habit**: Select smoking status
   - **Alcohol Consumption**: Select drinking habits
   - **Tobacco/Betel Chewing**: Select chewing tobacco usage

2. Click the **"Submit Demographics"** button

3. You should see: ✓ Demographics saved successfully!

#### Step 2: Upload Oral Cavity Image

1. After demographics submission, the image upload button will be enabled
2. Click **"Upload Image"** button
3. Select an oral cavity image from your computer (supports: JPG, PNG, etc.)
4. The system will automatically:
   - Load and preprocess the image
   - Run the AI model prediction
   - Generate Grad-CAM visualization
   - Display three images:
     - Original Image
     - Grad-CAM Heatmap
     - Grad-CAM Overlay with Prediction

#### Step 3: Download PDF Report

1. After image processing, a PDF report will be automatically generated
2. Click the download link that appears: **"Click here to download: [filename].pdf"**
3. The report includes:
   - Patient demographics
   - Model prediction (CANCER/NON-CANCER)
   - Confidence scores
   - Grad-CAM visualization
   - Risk assessment
   - Clinical recommendations

## Model Information

- **Architecture**: ResNet50 with custom classification head
- **Input Size**: 224 × 224 pixels
- **Output**: Binary classification (Cancer vs Non-Cancer)
- **Preprocessing**: 
  - Resize to 224×224
  - Normalization with ImageNet statistics
  - RGB color space

## Visualization

The system uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to highlight regions in the image that the model focuses on when making predictions. Red/yellow areas indicate regions of high importance to the prediction.

## Troubleshooting

### Issue: Widgets not displaying

**Solution**: 
- Make sure you've run the cell (not just viewing the code)
- Restart the kernel and run the cell again
- Verify ipywidgets is installed: `pip install ipywidgets`
- In Jupyter Lab, install the extension: `jupyter labextension install @jupyter-widgets/jupyterlab-manager`

### Issue: CUDA/GPU errors

**Solution**: 
- The system automatically falls back to CPU if CUDA is not available
- For CPU-only systems, this is normal and expected
- Processing may take slightly longer on CPU

### Issue: Model file not found

**Solution**: 
- Ensure `best_model_224_resnet50.pt` is in the same directory as `main.ipynb`
- Check the file name matches exactly (case-sensitive)

### Issue: Import errors

**Solution**: 
- Verify all packages are installed: `pip install -r requirements.txt`
- Check you're using the correct Python environment
- Try upgrading pip: `pip install --upgrade pip`

### Issue: PDF generation fails

**Solution**: 
- Check that `report.py` is present in the same directory
- Verify reportlab is installed: `pip install reportlab`
- Ensure write permissions in the directory

### Issue: GEMINI_API_KEY error

**Solution**: 
- Ensure you've created a `.env` file in the project directory
- Verify the `.env` file contains: `GEMINI_API_KEY=your_actual_api_key`
- Check that your API key is valid and active
- Make sure there are no extra spaces or quotes around the API key
- Restart the Jupyter kernel after creating/modifying the `.env` file

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all installation steps were completed
3. Ensure all required files are present in the directory

## License

This project is for medical research and educational purposes. Always consult with qualified healthcare professionals for actual medical diagnoses.

---

**⚠️ Disclaimer**: This system is a diagnostic aid tool and should not be used as the sole basis for medical decisions. Always consult with qualified healthcare professionals for proper diagnosis and treatment.
