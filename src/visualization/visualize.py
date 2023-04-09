import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
import os

def num_customers_category(con):

    """
    
    This function performs a query: Number of customers by job category then plots the result and saves the plot as a PNG file.
    
    Args:
        con (duckdb.Connection): DuckDB connection
        
    Returns:
        None

    """

    q_jobs = """
            SELECT job, COUNT(*) as num_customers
            FROM term_deposit_marketing
            GROUP BY job
            ORDER BY num_customers DESC;
            """
    
    # Execute the query
    df_jobs = con.execute(q_jobs).fetchall()
    df_jobs = pd.DataFrame(df_jobs, columns=['job', 'num_customers'])

    plt.figure(figsize=(10, 5))
    sns.barplot(x='num_customers', y='job', data=df_jobs)
    plt.title('Number of customers by job')
    
    # Save the plot
    plt.savefig(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'reports', 'figures', 'num_customers_category.png')))

if __name__=="__main__":

    # Set the path to the CSV file
    csv_path = os.path.abspath(os.path.join(os.getcwd(), '..','..', 'data', 'raw', 'term-deposit-marketing-2020.csv'))

    # Read the CSV file using pandas
    data = pd.read_csv(csv_path)

    # Create an in-memory DuckDB connection
    con = duckdb.connect(database=':memory:', read_only=False)

    # Register the pandas DataFrame as a DuckDB table
    con.register('term_deposit_marketing', data)

    # Execute the query
    num_customers_category(con)

    # Don't forget to close the connection
    con.close()