import sqlite3

db_path = "databases/HMDB_Serum_v01.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # First query to show column names
    print("=== Column Names from multiplets table (LIMIT 3) ===")
    cursor.execute("SELECT * FROM multiplets LIMIT 3;")
    columns = [description[0] for description in cursor.description]
    print("Columns:", columns)
    
    rows = cursor.fetchall()
    for i, row in enumerate(rows, 1):
        print(f"Row {i}: {row}")
    
    print("\n=== Query Results for metabolites with 'lanin' ===")
    query = """SELECT m.name, mp.ppm_center, mp.pattern, mp.protons, mp.j_coupling 
               FROM metabolites m 
               JOIN multiplets mp ON m.id=mp.metabolite_id 
               WHERE m.name LIKE '%lanin%' 
               ORDER BY mp.ppm_center;"""
    
    cursor.execute(query)
    columns = [description[0] for description in cursor.description]
    print("Columns:", columns)
    
    rows = cursor.fetchall()
    if not rows:
        print("No rows returned.")
    else:
        for i, row in enumerate(rows, 1):
            print(f"Row {i}: {row}")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
