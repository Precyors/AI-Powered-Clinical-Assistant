# Cleaning functions for txt and csv files with batch processing for csv files
def clean_txt_file(txt_file):
    doc_name = Path(txt_file).stem

    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
    
    cleaned_text = text.capitalize()
    
    insert_page(doc_name,
            len(cleaned_text),
            len(cleaned_text.split(" ")),
            len(list(nlp_2(cleaned_text).sents)),
            len(nlp(cleaned_text)),
            cleaned_text)
    return None


def clean_csv_file(csv_file, batch_size=25):
    doc_name = Path(csv_file).stem
    
    df = pd.read_csv(csv_file)

    df.dropna(inplace=True)  # drop rows with any NaN values
    df = df.select_dtypes(include=[object])  # select only string columns

    def clean_text(row):
        new_list = []
        for col in row.index:  # iterate through columns
            new_list.append(f"{col.capitalize()}: {row[col]}")  
        return ". ".join(new_list)
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        cleaned_text = batch_df.apply(clean_text, axis=1).to_list()
        cleaned_text = ' '.join(cleaned_text)
    
        insert_page(doc_name,
                    len(cleaned_text),
                    len(cleaned_text.split(" ")),
                    len(list(nlp_2(cleaned_text).sents)),
                    len(nlp(cleaned_text)),
                    cleaned_text)
    return None


