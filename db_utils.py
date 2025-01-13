import psycopg2
from psycopg2 import sql
import os

def create_connection_string():
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'password')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')

    db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}"

    return db_url


def create_database(db_name):
    conn_string = create_connection_string()

    try:

        conn = psycopg2.connect(conn_string)
        conn.autocommit = True
        cursor = conn.cursor()
        try:
            cursor.execute(
                sql.SQL("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s"),
                [db_name]
            )
            if cursor.fetchone():
                print(f"Database {db_name} already exists.")
            else:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                )
                print(f"Database {db_name} created successfully.")
        finally:
            cursor.close()
            conn.close()

    except psycopg2.Error as e:
        print(f"Error: {e}")
        if e.pgcode == '42P04':  # Database already exists
            print("Database already exists.")
        else:
            print("An error occurred while creating the database.")


def create_table(table_name):
    print(f'Starting creation of table: {table_name}')

    create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id BIGSERIAL PRIMARY KEY,
                embedding VECTOR(1536),
                document TEXT,
                metadata JSONB
            );
            """

    conn = None
    cur = None
    try:

        conn = psycopg2.connect(
            database='vector_db',
            user='postgres',
            password='password',
            host='localhost',
            port=5432
        )
        cur = conn.cursor()
        cur.execute(create_table_query)

        conn.commit()
        print(f'Table {table_name} created successfully')

    except psycopg2.Error as e:
        print(f'Database error occurred: {str(e)}')
        if conn:
            conn.rollback()
        raise

    except Exception as e:
        print(f'Unexpected error occurred: {str(e)}')
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            print('Database connection closed')

def get_db_connection_string(db_name):
    conn_string = create_connection_string()

    return f"{conn_string}/{db_name}"



create_table('rag_table')
