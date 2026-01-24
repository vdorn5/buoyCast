R# Copilot Instructions for buoyCast

## Project Overview
buoyCast develops wind-driven wave forecasting models using physics-informed machine learning. The project integrates physical wave dynamics with ML to predict wave heights, periods, and directions from wind data, ensuring predictions respect conservation laws and physical constraints.

## Big Picture Architecture
- **Data Ingestion**: Processes buoy measurements and meteorological wind fields (e.g., from NOAA or ECMWF).
- **Physics Engine**: Implements wave evolution equations like the wave action equation, incorporating wind input, nonlinear interactions, and dissipation.
- **ML Models**: Uses neural networks or other ML approaches with physics-informed constraints (e.g., PINNs - Physics-Informed Neural Networks) to learn from data while obeying physics.
- **Forecasting Pipeline**: Combines physics-based propagation with data-driven corrections for real-time predictions.

Key design decisions: Physics-informed approach reduces data requirements and improves extrapolation beyond training conditions, critical for oceanographic applications where labeled data is scarce.

## Developer Workflows
- **Environment Setup**: Use Docker with `docker-compose up -d` to build and start the container in the background, then `docker-compose exec buoycast bash` to enter the interactive shell. Ensure NVIDIA drivers are installed on the host for GPU access (runtime: nvidia).
- **Prototyping**: Start with Jupyter notebooks in `notebooks/` for data exploration and model development.
- **Training**: Run training scripts in `src/algorithms/` with GPU support via PyTorch/CUDA.
- **Evaluation**: Use evaluation scripts in `src/algorithms/` or `tests/` to compare predictions against buoy observations, focusing on metrics like RMSE for significant wave height.
- **Debugging**: Profile memory usage in physics computations; use TensorBoard for ML training visualization.

## Running the Project
- Build and start the container: `docker-compose up -d`
- Enter the interactive shell: `docker-compose exec buoycast bash`
- Stop the container: `docker-compose down`

## Project-Specific Conventions
- **Code Structure**: Follow `src/` for core modules, `data/` for datasets, `models/` for saved weights, `notebooks/` for exploratory work.
- **Physics Integration**: Always validate ML outputs against analytical physics solutions for simple cases (e.g., deep water waves).
- **Data Handling**: Store time series as xarray DataArrays for coordinate-aware operations; use zarr for large datasets.
- **Versioning**: Tag releases with physics validation milestones (e.g., "v1.0-physics-validated").

## Integration Points
- **External Dependencies**: Relies on libraries like PyTorch for autodiff in physics constraints, xarray for geospatial data, and scipy for numerical solvers.
- **Cross-Component Communication**: Data flows from ingestion → physics preprocessing → ML training → prediction; use Dask for parallel processing of large grids.

## Key Files/Directories
- `requirements.txt`: Core dependencies (numpy, scipy, pytorch, xarray, etc.)
- `docker/`: Docker setup files (Dockerfile, docker-compose.yml)
- `.github/workflows/`: CI/CD workflows for automated testing and builds
- `src/physics/`: Physics equation implementations
- `src/algorithms/`: ML algorithms, model definitions, and training loops
- `src/data/`: Data processing, ingestion, and preprocessing code
- `src/utils/`: Shared utilities and helper functions
- `notebooks/`: Jupyter notebooks for data exploration and prototyping
- `data/`: Datasets, buoy measurements, and wind fields
- `models/`: Saved model weights and checkpoints
- `tests/`: Unit tests and validation scripts