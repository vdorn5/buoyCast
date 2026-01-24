# buoyCast

[![CI](https://github.com/vdorn5/buoyCast/actions/workflows/docker-build.yml/badge.svg)](https://github.com/vdorn5/buoyCast/actions/workflows/docker-build.yml)

Wind-Driven Wave Forecasting with Physics-Informed Machine Learning

## Overview

buoyCast is a project that develops physics-informed machine learning models for predicting wind-driven ocean wave characteristics, such as wave heights, periods, and directions. By integrating physical wave dynamics with ML techniques, the project ensures predictions respect conservation laws and physical constraints, making it suitable for oceanographic applications where labeled data is scarce.

Key features:
- **Physics-Integrated ML**: Uses Physics-Informed Neural Networks (PINNs) to enforce physical laws during training.
- **Data Handling**: Processes buoy measurements and meteorological wind fields using xarray for efficient geospatial operations.
- **Scalable Training**: Supports GPU acceleration via PyTorch and CUDA.
- **Modular Architecture**: Organized into data processing, physics engines, ML algorithms, and evaluation components.

## Installation

### Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with drivers (for GPU acceleration; optional for CPU-only runs)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/vdorn5/buoyCast.git
   cd buoyCast
   ```

2. Build and start the development environment:
   ```bash
   docker-compose up -d
   ```

3. Enter the container:
   ```bash
   docker-compose exec buoycast bash
   ```

Dependencies are automatically installed from `requirements.txt` during the Docker build.

## Usage

### Development Workflow
- **Prototyping**: Use Jupyter notebooks in `notebooks/` for data exploration and model development.
- **Training**: Run scripts in `src/algorithms/` for model training.
- **Evaluation**: Use scripts in `tests/` or `src/algorithms/` to evaluate against buoy data.

Example: To run a training script (inside the container):
```bash
python src/algorithms/train.py
```

### Running Notebooks
Notebooks can be executed for testing or exploration:
```bash
jupyter notebook notebooks/
```

## Project Structure

```
buoyCast/
├── .github/
│   ├── copilot-instructions.md  # AI coding guidelines
│   └── workflows/               # CI/CD pipelines
├── docker/                      # Docker setup
├── src/                         # Source code
│   ├── algorithms/              # ML models and training
│   ├── data/                    # Data processing
│   ├── physics/                 # Physics equations
│   └── utils/                   # Shared utilities
├── notebooks/                   # Jupyter notebooks
├── data/                        # Datasets
├── models/                      # Saved weights
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker orchestration
└── README.md
```

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Make changes and add tests.
4. Run tests: `pytest` (inside the container).
5. Submit a pull request.

Please ensure code follows PEP 8 and includes physics validation for ML outputs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch, xarray, and other open-source tools.
- Inspired by advancements in physics-informed machine learning for environmental modeling.
