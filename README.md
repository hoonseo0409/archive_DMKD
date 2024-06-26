# Overview

Welcome to the code repository for our enrichment learning model, designed for traffic prediction within Intelligent Transportation Systems (ITS). Our model is dedicated to harnessing the full potential of both static (spatial) and dynamic (temporal) data, addressing the pressing need for safer, more efficient travel.
Our approach introduces a novel enrichment learning model that adeptly transforms disparate static and dynamic traffic data into a cohesive, fixed-length vector. This enriched representation allows traditional machine learning models to leverage the complete spectrum of traffic data, significantly boosting the accuracy of traffic congestion predictions.

## Repository Structure

- **app.py**: The entry point of the project, for the execution of the enrichment learning model along with other components.
- **enrichment.py**: The code of our enrichment model, containing the `Enrichment` class which processes data, executes the learning algorithm, and outputs results including performance metrics and visualizations into the `./out` directory.
- **helpers.py**: It is used for miscellaneous utilities and visualizations.
- **other_models_tester.py**: Contains benchmark models for comparative analysis.

## Getting Started

1. **Python Installation**: Ensure Python 3.7.3 is installed on your system. [Python Download](https://www.python.org)
2. **Pipenv Installation**: Use Pipenv for managing project-specific dependencies and virtual environments. [Pipenv Documentation](https://pipenv.pypa.io/en/latest/)

## Setup

1. Install all required Python packages listed in `./Pipfile` using Pipenv. Activate your virtual environment to ensure dependencies are properly managed.
2. Prepare your environment:
   - Place the `data` folder containing the Austin traffic dataset at the project root.
   - Create an `out` directory at the root to store output files.

## Data

You can access the speed measurements and traffic camera images data from Austin Department of Transportation at (https://data.mobility.austin.gov). From the downloaded data, `app.py` will generate the static and dynamic data using `load_austin` function.

## Execution

With the setup complete, run `app.py` to initiate the enrichment learning model. `app.py` requires two essential input data in numpy array format; static_data and dynamic_data which can be generated by `load_dataset.py`.

## Contributing

We encourage and appreciate contributions that enhance and expand the capabilities of our framework. Feel free to fork this repository and initiate a pull request to share your work.