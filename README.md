# UK Parliamentary Motions: Topic & Importance Classifier

[cite_start]**University of Greenwich - Applied Machine Learning** [cite: 183, 186]

## Overview
[cite_start]This repository contains the implementation, evaluation, and ethical assessment of a machine learning system designed to automatically analyze parliamentary motions from the UK House of Commons[cite: 271, 272]. [cite_start]The project explores whether machine learning can reliably support large-scale public policy analysis by predicting both the topic and the subjective priority of political texts[cite: 271, 291].

## Key Features & Tasks

* **Task 1: Topic Classification**
  [cite_start]A supervised machine learning pipeline that classifies parliamentary motions into six specific policy areas: terrorism and security, welfare and benefits, local government, Scotland devolution, climate and environment, and education[cite: 22]. [cite_start]Using Term Frequency-Inverse Document Frequency (TF-IDF) encoding (incorporating both unigrams and bi-grams) and a tuned Support Vector Machine (LinearSVC), the final model achieved an overall accuracy of 80%[cite: 47, 61, 73].

* **Task 2: Motion Priority Prototype**
  [cite_start]A prototype algorithm designed to identify which motions should be prioritized for further analysis[cite: 291]. [cite_start]This involved manually reviewing and labeling a subset of 100 motions as either "important" or "not important" based on perceived societal impact[cite: 109, 110]. [cite_start]A Logistic Regression model was trained on this custom dataset, successfully outperforming the random baseline by achieving a 60% accuracy rate[cite: 8, 128].

## Ethical Considerations & Data Hazards
[cite_start]A core component of this project is the critical evaluation of the ethical risks associated with deploying automated prioritization tools in a political context[cite: 94]. The project utilizes Data Hazards labels to identify and document:
* [cite_start]**Measurement Bias:** Due to the inherently subjective nature of defining motion "importance"[cite: 102].
* [cite_start]**Representation Bias:** Arising from the limited size and scope of the dataset[cite: 102].
* [cite_start]**Aggregation Bias:** The risk of treating complex, context-heavy parliamentary debates uniformly[cite: 102].
[cite_start]The findings strongly recommend using such models exclusively as decision-support tools with rigorous human oversight[cite: 104, 105].

## Technologies Used
* [cite_start]**Language:** Python [cite: 227]
* [cite_start]**Libraries:** Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `LinearSVC`) [cite: 155, 158, 160]
* [cite_start]**Techniques:** TF-IDF text encoding, Hyperparameter Tuning, Stratified Data Splitting, Data Preprocessing [cite: 34, 42, 64]

## Repository Structure
* [cite_start]`code_comp1804.ipynb` (or `.py`): The main source code detailing the implementation steps for data cleaning, text encoding, model building, and evaluation[cite: 227, 230].
* [cite_start]`report_comp1804.pdf`: A comprehensive technical report discussing the exploratory data analysis, experimental design, and ethical frameworks[cite: 225, 336].
* [cite_start]Original Dataset & Labeled Data: Includes the base motions dataset alongside the newly generated `motion_priority` labels[cite: 231, 234].

## Academic Integrity Notice
This project was developed as a coursework assignment. [cite_start]If you are a current student, please review your institution's academic integrity policies regarding referencing and code reuse[cite: 196, 203].