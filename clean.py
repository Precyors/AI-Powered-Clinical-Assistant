import pandas as pd
import re



def clean_data(data):
    """
    Clean data by removing duplicates and empty rows,
    convert textx to lowercase , and tokenize sentence columns.

    Return: Cleaned dataframe
    """
    try:

        # drop duplicate values
        df_cleaned = data.dropna(how='any')

        # delete rows with dup values
        df_cleaned = data.drop_duplicates()

        # convert columns to lowercase

        df_cleaned['Name'] = df_cleaned['Name'].str.lower()
        df_cleaned['Symptoms'] = df_cleaned['Symptoms'].str.lower()
        df_cleaned['Treatments'] = df_cleaned['Treatments'].str.lower()

        # Define a simple tokenization function
        def simple_tokenize(text):
            if pd.isna(text):
                return None
            # Use a regular expression to find all words
            return re.findall(r'\b\w+\b', text)

        # Apply the tokenization to the relevant columns
        df_cleaned['Symptoms_Tokenized'] = df_cleaned['Symptoms'].apply(simple_tokenize)
        df_cleaned['Treatments_Tokenized'] = df_cleaned['Treatments'].apply(simple_tokenize)

        return df_cleaned
    
    except KeyError as e:
        print(f"Error: A required column was not found. Please check your DataFrame for column: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data cleaning: {e}")
        return None


if __name__ == "__main__":
    try:
        # This part of the code needs to be adjusted based on your file path
        raw_data = pd.read_csv(r'C:\Users\USER\Documents\AI_Clinical_Assistant\data\Diseases_Symptoms.csv')
        cleaned_df = clean_data(raw_data)
        if cleaned_df is not None:
            print("Data cleaning completed successfully.")
            # You can save the cleaned data here
            cleaned_df.to_csv(r"C:\Users\USER\Documents\AI_Clinical_Assistant\data\clean_data", index=False)
            print(cleaned_df.head())
    except FileNotFoundError:
        print("Error: The CSV file was not found. Please check the file path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")