import os
import pg8000
import psycopg2
import ssl
from datetime import datetime
from psycopg2 import pool

db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_ssl = os.getenv("DB_SSL")
max_open_conn = os.getenv("MAX_OPEN_CONNECTIONS")

max_conn_age = 120  # Max age in seconds, 3600 seconds = 1 hour
# Maintain a dictionary to store connection creation times
connection_pool = None
connection_times = {}

# Function to create a new connection
def create_new_connection():
    global connection_pool
    if connection_pool is None:
        if db_ssl == 'require':
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        else:
            ssl_context = None

        connection_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=max_open_conn,
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
            sslmode=db_ssl,
            sslcert=None,
            sslkey=None,
            sslrootcert=None
        )

    new_conn = connection_pool.getconn()
    connection_times[new_conn] = datetime.now()
    return new_conn

# Function to get a connection
def get_connection():
    current_time = datetime.now()
    
    # List to store connections to close
    to_close = []
    
    for conn, creation_time in connection_times.items():
        age = (current_time - creation_time).seconds
        try:
            # Check if the connection is closed
            if conn.closed:
                to_close.append(conn)
            elif age > max_conn_age:
                # Add connection to the list to close
                to_close.append(conn)
        except Exception as e:
            # If there's an error, assume the connection is bad and close it
            to_close.append(conn)
    
    # Close the old connections
    for conn in to_close:
        connection_pool.putconn(conn, close=True)
        del connection_times[conn]
    
    # Check if there are still valid connections
    for conn, creation_time in connection_times.items():
        return conn  # Return the first valid connection found
    
    # If no valid connection is found, create a new one
    new_conn = create_new_connection()
    
    #print("Database connected successfully to ", db_host)
    return new_conn

def connect_to_database():
    if db_ssl == 'require':
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    else:
        ssl_context = None

    conn = pg8000.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
        port=db_port,
        ssl_context=ssl_context
    )
    return conn