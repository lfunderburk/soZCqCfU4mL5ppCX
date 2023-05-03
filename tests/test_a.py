import duckdb
import pandas as pd
from src.visualization.visualize import num_customers_category

def test_num_customers_category():
    # Set up the test data
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE customers (job VARCHAR(20), age INT)")
    con.execute("INSERT INTO customers VALUES ('admin.', 30)")
    con.execute("INSERT INTO customers VALUES ('admin.', 40)")
    con.execute("INSERT INTO customers VALUES ('technician', 35)")
    
    # Define the expected output
    expected_output = pd.DataFrame({"job": ["admin.", "technician"], "num_customers": [2, 1]})
    
    # Call the function
    output = num_customers_category(con, "SELECT job, COUNT(*) as num_customers FROM customers GROUP BY job", ["job", "num_customers"])
    
    # Check the output
    pd.testing.assert_frame_equal(output, expected_output)

    # Close the connection
    con.close()