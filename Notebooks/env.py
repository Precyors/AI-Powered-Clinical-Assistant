# Contains the path to the data folder and the function to get the list of files
def get_file_list(data_path=DATAFILES_PATH):
    data_path = Path(data_path)
    dirs = []
    files = []

    for item in data_path.rglob("*"):
        if item.is_dir():
            dirs.append(item)
        else:
            files.append(item)

    all_files = [str(i) for i in files if not (Path(i).is_dir() or str(i).endswith(".zip"))]
    pdf_files = [str(i) for i in all_files if str(i).endswith(".pdf")]
    txt_files = [str(i) for i in all_files if str(i).endswith(".txt")]
    csv_files = [str(i) for i in all_files if str(i).endswith(".csv")]
    
    return all_files, pdf_files, txt_files, csv_files