import os
import mysql.connector

class Database:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306))
        )

    def execute(
        self,
        query,
        params=None,
        fetchone=False,
        fetchall=False
    ):
        cursor = None
        try:
            # ✅ dictionary=True supaya hasil berupa dict
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute(query, params or ())

            # SELECT
            if fetchone:
                return cursor.fetchone()

            if fetchall:
                return cursor.fetchall()

            # INSERT / UPDATE / DELETE
            self.conn.commit()
            return True

        except Exception as e:
            # rollback jika terjadi error
            try:
                self.conn.rollback()
            except:
                pass

            print(f"[DB ERROR] {e}")
            return None

        finally:
            if cursor:
                cursor.close()

    def close(self):
        self.conn.close()