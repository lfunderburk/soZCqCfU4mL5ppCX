Success rate improvement
==============================

## Contents

0. [Usage](#pipeline-usage)
1. [Data Description](#data-description)
2. [Goals](#goals)
3. [Exploratory Data Analysys](#exploratory-data-analysis)
4. [Building a model](#building-a-model-to-answer-the-questions)
5. [Results](#results)

## Pipeline usage

This pipeline is built using [Ploomber](https://ploomber.io/). 

Follow the instructions below:

1. Ensure you have a virtual environment. 
2. Activate virtual environment

```
git clone https://github.com/lfunderburk/soZCqCfU4mL5ppCX.git
cd soZCqCfU4mL5ppCX
pip install -r requirements.txt
```

Wait until installation completes. The to execute pipeline, simply run

```
ploomber build
```

It should look as follows

```
name           Ran?      Elapsed (s)    Percentage
-------------  ------  -------------  ------------
visualize      True          4.60473       18.5278
train_model    True         15.7014        63.1768
predict_model  True          4.54695       18.2953
```

Fetch results from the `models` and `reports` folders.

## Data Description

Data Description:

The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.


### Attributes:

| Column	|Description|
|-|-|
|age	|Age of customer (numeric)|
|job	|Type of job (categorical)|
|marital	|Marital status (categorical)|
|education	|Education level (categorical)|
|default	|Has credit in default? (binary)|
|balance	|Average yearly balance, in euros (numeric)|
|housing	|Has a housing loan? (binary)|
|loan	|Has personal loan? (binary)|
|contact	|Contact communication type (categorical)|
|day	|Last contact day of the month (numeric)|
|month	|Last contact month of the year (categorical)|
|duration	|Last contact duration, in seconds (numeric)|
|campaign	|Number of contacts performed during this campaign and for this client (numeric, includes last contact)|

### Output (desired target):

$y$ - has the client subscribed to a term deposit? (binary)

## Goals:

- Predict if the customer will subscribe (yes/no) to a term deposit (variable y)
- Find customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.
- What makes the customers buy? 

## Exploratory data analysis

### Find the number of customers who subscribed to a term deposit (y = 'yes') by job category

![](./reports/figures/num_subscribers_job.png)

It appears that the group of customers subscribed to a term deposit were predominantly workers in management positions, followed by workers in blue-collar positions, technicians and workers in admin roles. 

Representation decreases dramatically for employees in services, or employees who are either retired, self employed, unemployed, a student, or a housemaid. 

This suggests that subscribed customers may enjoy higher income stability. 

### Find the number of customers who subscribed to a term deposit (y = 'yes') by marital status:

![](./reports/figures/num_subscribers_marital.png)

Customers whose marital status is married, also subscribed to a term deposit at higher rate than single or divorced customers. It is worth noting that there were at least 3 times as many single customers that divorced customers. 

### Find the number of customers who subscribed to a term deposit (y = 'yes') by education level:

![](./reports/figures/num_subscribers_education.png)

The two most prominent levels of education completed by customers subscribed to a term deposit was secondary education, followed by tertiary education (university, trade schools, colleges).

### Find the number of customers who subscribed to a term deposit (y = 'yes') by housing loan status:

![](./reports/figures/num_subscribers_housing.png)

The breakdown of customers with a term deposit is broken down quite similarly, when breaking them down by housing loan status (did have a loan status, or did not have a loan status).

### Find the number of customers who subscribed to a term deposit (y = 'yes') by age bracket and marital status:

![](./reports/figures/num_customers_m_bracket.png)

Customers between 30 and 39 appeared to represent a large percentage of customers with a term deposit, whether they were single or married. Of those customers that were single, customers aged 20 to 29 were the next most represented group. Of those customers that were married, those aged 40-59 were the next most represented group. 

## Building a model to answer the questions:

- Predict if the customer will subscribe (yes/no) to a term deposit (variable y)
- Find customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.
- What makes the customers buy? 

### Data preprocessing

No missing information was found. Data contained both numerical and categorical attributes. 

### Pipeline development


![](/models/pipeline_diagram.png)

## Results



```
Best parameters: {'undersampler__sampling_strategy': 'not majority', 'preprocessor': ColumnTransformer(transformers=[('num', StandardScaler(),
                                 ['age', 'balance', 'day', 'duration',
                                  'campaign']),
                                ('cat', OneHotEncoder(handle_unknown='ignore'),
                                 ['job', 'marital', 'education', 'default',
                                  'housing', 'loan', 'contact', 'month'])]), 'classifier__subsample': 0.8, 'classifier__scale_pos_weight': 1, 'classifier__reg_lambda': 0, 'classifier__reg_alpha': 1, 'classifier__n_estimators': 75, 'classifier__max_depth': 7, 'classifier__learning_rate': 0.1, 'classifier__colsample_bytree': 0.5}
                                  
Cross-validated accuracy score on training data: 0.9296

Weighted average F1 score 0.9311178658922542
Macro average F1 score 0.7593392488747186
Micro average F1 score 0.927

Accuracy: 0.927
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.95      0.96     11131
           1       0.50      0.64      0.56       869

    accuracy                           0.93     12000
   macro avg       0.73      0.79      0.76     12000
weighted avg       0.94      0.93      0.93     12000
        
```

### Feature importances

![](/reports/figures/feature-importances.png)


### High priority customers

The following breakdown in attibutes of customers was observed for customers deemed "high priority" using a threshold of 70%


![](/reports/figures/customer-segmentation.png)

## Conclusion

#### Find customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.

Customers holding a management, technician, admin roles, or working in services and blue collar jobs should be prioritized. Look for customers that completed secondary education, and are either maried or single. Prioritize customers that have not defaulted, and do not have a loan. 

#### What makes the customers buy? 

Customers that do not own housing or loans, and that spend time speaking with representatives about the program are most influenced to buy. Increase time speaking with them through various means about the investment opportunity. 

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
