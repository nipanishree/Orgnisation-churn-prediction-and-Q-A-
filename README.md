# Customer Churn Prediction and Q&A System

A comprehensive machine learning project combining customer churn prediction with a Retrieval-Augmented Generation (RAG) based question answering system.

## Project Structure

```
ProjectsOrgnisation-churn-prediction-and-Q-A/
├── .github/workflows/          # CI/CD workflows
├── data/
│   ├── raw/                    # Raw data files
│   ├── interim/                # Intermediate processed data
│   └── processed/              # Final processed data
├── notebooks/                  # Jupyter notebooks for exploration
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model implementations
│   ├── rag/                    # RAG system components
│   └── utils/                  # Utility functions
├── api/                        # FastAPI application
├── tests/                      # Unit and integration tests
├── documents/
│   └── knowledge_base/         # Knowledge base for RAG
├── models/                     # Trained models
├── configs/                    # Configuration files
├── pyproject.toml              # Project dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Features

- **Churn Prediction**: Machine learning models to predict customer churn
- **Q&A System**: RAG-based question answering using knowledge base
- **REST API**: FastAPI-based API for model serving
- **Data Pipeline**: Complete ETL pipeline for data processing
- **Testing**: Comprehensive test suite

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ProjectsOrgnisation-churn-prediction-and-Q-A
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Copy environment variables:
```bash
cp .env.example .env
```

### Usage

#### Running the API

```bash
uvicorn api.main:app --reload
```

#### Running Tests

```bash
pytest tests/ -v
```

#### Running Notebooks

```bash
jupyter notebook
```

## Development

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run checks:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Ensure tests pass
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contact

For questions or issues, please create an issue in the repository.
