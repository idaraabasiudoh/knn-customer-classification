# KNN_classifier_telecommunication_test_project
Labels telecommunication customer base to respective groups to determine service type required for each customer. 

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Contributions](#contributions)
- [Acknowledgments](#acknowledgments)
- [Change Log](#change-log)
- [License](#license)

## Introduction
This repository contains a machine learning project focused on classifying customers for a telecommunication company using the K-Nearest Neighbors (KNN) algorithm. The project leverages Python and popular data science libraries such as scikit-learn, pandas, and matplotlib.

## Objectives
The primary objectives of this project are:
- To implement a K-Nearest Neighbors model using scikit-learn to classify customers.
- To train, test, and evaluate the model on a dataset of customer demographics and usage patterns.
- To explore the relationship between different customer features and their classification.

## Dataset
The dataset used in this project contains demographic data and service usage patterns of customers, segmented into four groups. The target field, `custcat`, has four possible values that correspond to the four customer groups:
1. Basic Service
2. E-Service
3. Plus Service
4. Total Service

The dataset includes the following columns:
- `region`
- `tenure`
- `age`
- `marital`
- `address`
- `income`
- `ed`
- `employ`
- `retire`
- `gender`
- `reside`

[Dataset Source](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv)

## Installation
To run this project locally, you need to have Python installed along with the required libraries. You can install the necessary packages using the following command:

Clone the repository and install the necessary dependencies:

git clone https://github.com/idaraabasiudoh/knn-customer-classification.git
cd knn-customer-classification
pip install -r requirements.txt

## Usage
To use this repository, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/idaraabasiudoh/knn-customer-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd knn-customer-classification
    ```
3. Run the classification script:
    ```bash
    python knn-customer-classification.py
    ```

## Modeling
The modeling process involves the following steps:
1. **Data Exploration**: Understanding the dataset by visualizing and summarizing the data.
2. **Data Preparation**: Cleaning and splitting the data into training and testing sets.
3. **Model Training**: Using the training set to train a K-Nearest Neighbors model.
4. **Model Evaluation**: Evaluating the model using metrics such as accuracy.

## Evaluation
The performance of the model is evaluated using the test dataset. The key metrics used for evaluation include:

- **Accuracy**: This metric indicates how well the model's predictions match the actual classifications.

### Example Code
Here is an example of how to evaluate the model using these metrics:
```python
from sklearn.metrics import accuracy_score

# Assuming y_test contains the actual values and y_pred contains the predicted values
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
## Contributions
We welcome contributions from the community to improve this project. To contribute, please follow these steps:

1. **Fork the Repository**: Click the "Fork" button at the top right of the repository page to create a copy of this repository on your GitHub account.
2. **Clone the Repository**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/idaraabsiudoh/knn-customer-classification.git
    ```
3. **Create a New Branch**: Create a new branch for your feature or bug fix.
    ```bash
    git checkout -b feature-name
    ```
4. **Make Changes**: Make your changes to the codebase.
5. **Commit Your Changes**: Commit your changes with a clear and descriptive commit message.
    ```bash
    git commit -m "Description of your changes"
    ```
6. **Push to Your Branch**: Push your changes to your forked repository.
    ```bash
    git push origin feature-name
    ```
7. **Open a Pull Request**: Open a pull request to merge your changes into the main repository. Provide a detailed description of your changes in the pull request.

We appreciate your contributions and will review your pull request as soon as possible. Thank you for helping improve this project!

## Acknowledgments 
<a href="http://www.linkedin.com/in/idaraabasiudoh" target="_blank">Idara-Abasi Udoh</a>

Saeed Aghabozorgi

### Other Contributors
<a href="https://www.linkedin.com/in/joseph-s-50398b136/" target="_blank">Joseph Santarcangelo</a>

Azim Hirjani

## Change Log
| Date (YYYY-MM-DD) | Version | Changed By         | Change Description                |
|-------------------|---------|--------------------|-----------------------------------|
| 2024-07-02        | 2.2     | Idara-Absi Udoh    | Project completion                |
| 2020-11-03        | 2.1     | Lakshmi Holla      | Changed URL of the csv            |
| 2020-08-27        | 2.0     | Lavanya            | Moved lab to course repo in GitLab |

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
