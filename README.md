
# Titanic Dataset machine learning final project

This repository demonstrates a machine learning pipeline applied to the Titanic survival dataset.

The project covers:

- Data preprocessing and feature engineering
- Feature selection using Sequential Feature Selector (SFS)
- Model comparison and evaluation (Logistic Regression, Random Forest, SVM, KNN)
- Hyperparameter tuning with GridSearchCV
- Model saving and deployment using pickle
- A simple interactive Streamlit dashboard for predicting survival probability

The purpose of this project is to provide a complete end-to-end ML workflow, from data preprocessing to model deployment, as an example of practical machine learning implementation.

## Installation

Clone the repository:

```bash
  git clone git@github.com:erfantbtb/machine_learning_course_final_project.git
```

Cloning and installing python dependencies:
```bash
  cd machine_learning_course_final_project
  python3 -m venv ml_venv
  source ml_venv/bin/activate
  pip install -r requirements.txt
```

## Run Locally

After installation, for running the UI, we use streamlit CLI:

```bash
  streamlit run main.py
```
![User Interface](https://github.com/erfantbtb/machine_learning_course_final_project/blob/main/repo_images/ui.png)

## Authors

- [@erfantbtb](https://github.com/erfantbtb)
- [@ALIXxDN](https://github.com/ALIXxDN)
