Success rate improvement
==============================

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

### Goal(s):

- Predict if the customer will subscribe (yes/no) to a term deposit (variable y)
- Find customers who are more likely to buy the investment product. Determine the segment(s) of customers our client should prioritize.
- What makes the customers buy? 

## Exploratory data analysis

### Find the number of customers who subscribed to a term deposit (y = 'yes') by job category

![](./reports/figures/num_subscribers_job.png)

### Find the number of customers who subscribed to a term deposit (y = 'yes') by marital status:

![](./reports/figures/num_subscribers_marital.png)

### Find the number of customers who subscribed to a term deposit (y = 'yes') by education level:

![](./reports/figures/num_subscribers_education.png)

### Find the number of customers who subscribed to a term deposit (y = 'yes') by housing loan status:

![](./reports/figures/num_subscribers_housing.png)

### Find the number of customers who subscribed to a term deposit (y = 'yes') by age bracket and marital status:

![](./reports/figures/num_customers_m_bracket.png)


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
