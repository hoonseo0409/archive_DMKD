## This code is implementation of enrichment learning model.
This is used in experiments section of paper.

## Project Structure
* app.py: Entry point of this project. The other codes will be imported here and will be run.
* enrichment.py: Core codes to implement enrichment learning model. Class 'Enrichment' will takes data and experiments, and `fit` method will train the enrichment model with the given dataset, and save the results (scores and visualizations) under ./out directory.
* group_finder.py: Not important code, just utility class to find the group structure of snips data.
* helpers.py: It was originally used as utility and visualization helper codes, but now it was replaced by utilsforminds packages.
* other_models_tester.py: Codes for the benchmark comparison models. These comparison models will compete with enrichment learning model.

## How to run
* Install Python 3.7.3 (https://www.python.org) (For macos user, I recommend to install Python through brew).
* Install pipenv (https://pipenv.pypa.io/en/latest/) (For macos user, again I recommend to install pipenv through brew). Pipenv is environment manager, which will create independent Python dedicated to the specific project and manage the package dependencies for that project (similar to virtualenv).
* Install Git (https://git-scm.com).
* Install the required packages for this project. The packages are listed under `./Pipfile`. When you install packages, don't forget to create and activate (both can be done by `pipenv shell`) your local pipenv virtual environment by checking whether `pipenv --py` points to the correct Python path.
    * The public Python packages can be simply installed via `pipenv install` which will install all the packages listed under ./Pipfile.
    * utilsforminds is a custom package and should be installed via `pipenv install -e "path-to-utilsforminds"`.
* (Optional) You may need to install programming editor to work conveniently. I recommend to use vscode (https://code.visualstudio.com).
    * After installing vscode, you can link the pipenv Python to your project directory. At the left down corner, click the Python and set the path to pipenv local Python executable.
* Put the 'data' folder for Austin traffic dataset at the root.
* Put the 'outer_sources' folder for Python codes for visualization.
* Create an empty folder named 'out' at the root. 
* If everythings are setup correctly, you will be able to run `app.py`.