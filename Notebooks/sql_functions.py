# SQL functions for creating tables and inserting/extracting data
CONN = sqlite3.connect("../Database/rag_documents.db")

def create_tables():
    # Create a DB connector
    cursor = CONN.cursor()
    
    # Create unified table
    cursor.execute("DROP TABLE IF EXISTS documents")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_name TEXT,
        character_count INTEGER,
        word_count INTEGER,
        sentence_count INTEGER,
        token_count INTEGER,
        text TEXT
    )
    """)
    
    CONN.commit()

def insert_page(doc_name, character_count, word_count, sentence_count, token_count, text):
    cursor = CONN.cursor()
    cursor.execute("""
        INSERT INTO pages (doc_name, character_count, word_count, sentence_count, token_count, text)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (doc_name, character_count, word_count, sentence_count, token_count, text))
    CONN.commit()

def extract_pages():
    cursor = CONN.cursor()
    cursor.execute("SELECT * FROM pages")
    rows = cursor.fetchall()
    df = pd.DataFrame(extract_pages(), columns=['id', 
                                                'doc_name', 
                                                'char_count', 
                                                'word_count',
                                                'sent_count', 
                                                'token_count',
                                                'text']).set_index('id')
    return df

def clear_pages():
    cursor = CONN.cursor()
    cursor.execute("DELETE FROM pages")
    CONN.commit()