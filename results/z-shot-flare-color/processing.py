import os
import pandas as pd
import numpy as np

# WRAPPERS
def raise_flag(image_file, predicted_class, ground_truth, difficulty):
    if (predicted_class != "blue") & (predicted_class != "red"):
        interpreted_results.write("Error for image " + str(image_file) + ". Could not find a color.\n")
        return
    interpreted_results.write("Error for image " + str(image_file) + ". Incorrectly classified as: " + predicted_class + ". Was " + difficulty + " to tell.\n")

    return

def get_model_name_from_file(filename):
    filename = filename.removesuffix(".csv")
    return filename.split("-")[-1]

def write_confusion_matrix(confusion_matrix):
    TN = confusion_matrix["TN"]
    TP = confusion_matrix["TP"]
    FN = confusion_matrix["FN"]
    FP = confusion_matrix["FP"]
    
    interpreted_results.write("Confusion matrix for blue identification:\nTP: " + str(TP) + "    FN: "+ str(FN) + "\n")
    interpreted_results.write("FP: " + str(FP) + "    TN: " + str(TN) + "\n\n")

    return

def compute_F1_score(confusion_matrix):
    TN = confusion_matrix["TN"]
    TP = confusion_matrix["TP"]
    FN = confusion_matrix["FN"]
    FP = confusion_matrix["FP"]
    
    precision = TP / (TP + FP) if TP + FP != 0
    recall = TP / (TP + FN) if TP + FN != 0

    if precision + recall != 0
        return 2*precision*recall / (precision+recall)

    return None

def write_F1_score(confusion_matrix):
    F1 = compute_F1_score(confusion_matrix)

    if F1 is not None:
        interpreted_results.write("F1 score: " + str(F1) + "\n")

    return
    
def corr(confusion_matrix):
    j = np.ones(2, dtype=int)
    r = np.asarray([k for k in range(1, 3)])
    r2 = np.asarray([k**2 for k in range(1, 3)])
    
    n = np.dot(np.dot(np.transpose(j), confusion_matrix), j)
    sum_x = np.dot(np.dot(np.transpose(r), confusion_matrix), j)
    sum_y = np.dot(np.dot(np.transpose(j), confusion_matrix), r)
    sum_x2 = np.dot(np.dot(np.transpose(r2), confusion_matrix), j)
    sum_y2 = np.dot(np.dot(np.transpose(j), confusion_matrix), r2)
    sum_xy = np.dot(np.dot(np.transpose(r), confusion_matrix), r)

    if np.sqrt(n*sum_x2 - sum_x**2) * np.sqrt(n*sum_y2 - sum_y**2) != 0:
        return (n*sum_xy - sum_x*sum_y)/(np.sqrt(n*sum_x2 - sum_x**2) * np.sqrt(n*sum_y2 - sum_y**2))

    return None
    
def write_corr_coeff(confusion_matrix):
    corr_coeff = corr(confusion_matrix)

    if corr_coeff is not None:
        interpreted_results.write("r = " + str(corr_coeff) + "\n")

    return



# PROCESSING
# Create and open an interpreted results file
interpreted_results = open("results.txt", "w")

# Loading all files, only keep csv files
""" If u wanna do the whole directory
og_dir_list = os.listdir()
dir_list = og_dir_list.copy()

for filename in og_dir_list:
    file_ext = filename.split(".")[-1]
    if ".csv" not in file_ext:
        dir_list.remove(filename)
"""
# For working on 100 imgs
dir_list = ["100-imgs-34b.csv", "100-imgs-vicuna-13b.csv"]

# Treat all csv files
for result_csv in dir_list:
    model_name = get_model_name_from_file(result_csv)
    nb_imgs = result_csv.split("-")[0]
    interpreted_results.write("------------------ Testing for " + model_name + " on " + nb_imgs + "images " + "------------------\n\n")
    
    # If the flare is blue
    TP = 0
    FN = 0

    # If the flare is red
    TN = 0
    FP = 0

    results_df = pd.read_csv(result_csv)

    for row in range(results_df.shape[0]):
        ground_truth = results_df['ground_truths'][row]
        predicted_class = results_df['predictions'][row]

        if ground_truth == predicted_class:
            TP += 1 if ground_truth == "blue" else 0
            TN += 1 if ground_truth == "red" else 0

        else:
            FP += 1 if ground_truth == "red" else 0
            FN += 1 if ground_truth == "blue" else 0
            
            raise_flag(results_df['imgs'][row], predicted_class, ground_truth, results_df['difficulties'][row])

    interpreted_results.write("In the end, detected " + str(100 * TP/(TP + FN)) + "% of blue flares.\n")
    interpreted_results.write("In the end, detected " + str(100 * TN/(TN + FP)) + "% of red flares.\n")
    
    confusion_matrix = {"TP": TP, "FN": FN, "FP": FP, "TN": TN}
    write_confusion_matrix(confusion_matrix)
    write_F1_score(confusion_matrix)
    write_corr_coeff(np.asarray([confusion_matrix[key] for key in confusion_matrix]).reshape((2, 2)))

interpreted_results.close()
