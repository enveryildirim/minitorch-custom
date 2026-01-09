# Custom minitorch 
fork of [minitorch](https://github.com/minitorch/minitorch) repo for self educations

## Installation

1. **Install the minitorch package in development mode:**
```bash
pip install -e .
python verify_installation.py

```

2. **Install additional dependencies for the streamlit app:**
```bash
pip install -r requirements.extra.txt
```

## Requirements

https://github.com/chalk-diagrams/chalk
```bash
conda install -c conda-forge chalk-diagrams
```

## Running the Streamlit App

After installing the package, run the app with:
```bash
streamlit run project/app.py -- 0
```

The `-- 0` parameter specifies the module number (0-4 depending on which module you're working on).
