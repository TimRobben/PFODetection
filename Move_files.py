import pandas as pd


def copy_selected_cts(excel_path, source_folder, destination_folder,
                                 sheet_name=0, filename_column='Imaging pseudo ID',
                                 column1='CTA_HEART_PFO', column2='ECHOCARD_STRUCTURAL_ABN_INFO.Patent_foramen_ovale_PFO'):
    """
    Copies selected CT files from a source folder to a destination folder based on an Excel file.
    Only files where both column1 and column2 are 'yes' (case-insensitive) are copied.
    
    Parameters:
      excel_path (str): Path to the Excel (.xlsx) file.
      source_folder (str): Path to the folder containing the CT files.
      destination_folder (str): Path to the folder where selected CT files will be copied.
      sheet_name (str or int, optional): The sheet in the Excel file to read (default is the first sheet).
      filename_column (str, optional): The column name that contains the CT file names.
      column1 (str, optional): The first column to check for 'yes'.
      column2 (str, optional): The second column to check for 'yes'.
    """
    # Read the Excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # Filter rows where both column1 and column2 are 'yes' (ignoring case and extra spaces)
    # df_filtered = df[(df[column1].astype(str).str.strip().str.lower() == 'yes') & (df[column2].astype(str).str.strip().str.lower() == 'yes')]
    df_filtered = df[(df[column1].astype(str).str.strip().str.lower() == 'yes')]
    count = 0
    list =[]
    # Print the filenames for each filtered row
    for file_name in df_filtered[filename_column]:
        # print(file_name)
        list.append(file_name)
        list.sort()
        count+= 1
    print(list, count)
    
# Example usage:
copy_selected_cts('/scratch/tarobben/MTHdata_imagingID_with_scans.xlsx', '/scratch/tarobben/CT_scans_original', '/scratch/tarobben/only_PFO')
