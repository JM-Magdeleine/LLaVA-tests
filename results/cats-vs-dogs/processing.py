"""
PLACEHOLDER SCRIPT, NOT REAL PROCESSING SCRIPT
IMPORTED FROM RESTITUTION TEST PROCESSING
"""
import pandas as pd
import os
import calendar

# WRAPPERS
def get_model_name_from_file(filename):
    return filename.removesuffix(".csv")

results = open("test-results.txt", "w")
dir_items = os.listdir()
dir_items.sort()

# Ground truth
ground_truth = pd.read_csv("ground-truth.csv")

# Only retrieve csv files
for item in dir_items:
    if ".csv" not in item:
        dir_items.remove(item)
for item in dir_items:
    if "truth" in item:
        dir_items.remove(item)
dir_items.remove("processing.py~")

for data in dir_items:
    rights = 0
    model_name = get_model_name_from_file(data)
    results.write("-------------------- Results for " + model_name + " --------------------\n\n")

    # Actual results
    df = pd.read_csv(data)

    # Check for sizes
    nb_cols_gt = len(ground_truth.columns)
    nb_cols_test = len(df.columns)
    nb_rows_gt = len(ground_truth[ground_truth.columns[0]])
    nb_rows_test = len(df[df.columns[0]])
    rows_diff = nb_rows_gt - nb_rows_test
    cols_diff = nb_cols_gt - nb_cols_test
    
    if cols_diff != 0:
        results.write("Wrong number of columns: ")
        if cols_diff > 0:
            results.write("missed " + str(cols_diff) + " column" + ("" if cols_diff > 1 else "s") + ".\n")
        else:
            results.write("hallucinated " + str(abs(cols_diff)) + " column" + ("" if cols_diff < -1 else "s") + ".\n")

    if rows_diff != 0:
        results.write("Wrong number of rows: ")
        if rows_diff > 0:
            results.write("missed " + str(rows_diff) + " row"  + ("" if rows_diff > 1 else "s") + ".\n")
        else:
            results.write("hallucinated " + str(abs(rows_diff)) + " row"  + ("" if cols_diff < -1 else "s") + ".\n")

    # Compare values
    rows = min(nb_rows_gt, nb_rows_test)
    columns, len_columns = df.columns[1:], len(df.columns[1:])
    months = calendar.month_abbr[1:]
    
    for column in columns:
        for row in range(rows):
            if df[column][row] != ground_truth[column][row]:
                results.write("Error for '" + str(column) + "' value of " + months[row] + " should be: " + str(ground_truth[column][row]) + " was: " + str(df[column][row]) + "\n")
            else:
                rights += 1

    results.write("In the end, got " + str(100*rights/(rows*len(columns))) + "% of data points right\n\n\n")

    

results.close()
