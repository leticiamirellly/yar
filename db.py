import psycopg2
import numpy as np
import json

DB_CONNECTION_STRING = "host='localhost' port='5432' dbname='' user='' password=''"

# cur = conn.cursor()
# cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

# table_create_command = """
# CREATE TABLE IF NOT EXISTS embeddings (
#             id bigserial primary key, 
#             content text,
#             embedding vector(768)
#             );
#             """
# cur.execute(table_create_command)

# conn.commit()

print('Connected to PostgreSQL...')

class StoreResults:
    def __call__(self, batch):
        print("Storing batch...")
        try:
            with psycopg2.connect(DB_CONNECTION_STRING) as conn:
                # register_vector(conn)
                print("CONNECTED")
                with conn.cursor() as cur:
                    for text, embedding in zip(batch["text"], batch["embeddings"]):    
                        embedding_list = np.array(embedding).tolist()
                        cur.execute("INSERT INTO embeddings (content, embedding) VALUES (%s, %s) RETURNING id", (text, embedding_list))
                        conn.commit()
            # conn.close()
            return {}
        except Exception as e:
            print(f"Failed to insert: {text[:50]}... Error: {str(e)}")
        print("Batch stored successfully.")


class getContext:
    def __call__(self, batch):
        print("Storing batch...")
        try:
            with psycopg2.connect(DB_CONNECTION_STRING) as conn:
                print("CONNECTED")
                with conn.cursor() as cur:
                    embedding_array = np.array(batch).tolist()
                    cur.execute("""SELECT content FROM embeddings ORDER BY embedding <-> %s::vector LIMIT 5""",
                            (batch,))
                    rows = cur.fetchall()
                    return rows
        except Exception as e:
            print(e)
        # print("Batch stored successfully.")
