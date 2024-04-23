"""
Script for processing 1-shot raw test results.
Outputs a results.txt file containing:
    The confusion matrix,
    The F1 score,
    The r coefficient of confusion matrix diagonality,
for each test and each model.
"""
import os, json
import pandas as pd
import numpy as np

# WRAPPERS
def raise_flag(image_name, predicted_class, ground_truth):
    if (predicted_class != "yes") & (predicted_class != "no"):
        interpreted_results.write("Error for image " + str(image_name) + ". Could not find a classification.\n")
        return
    interpreted_results.write("Error for image " + str(image_name) + ". Incorrectly flagged: " + predicted_class + "\n")

    return

def get_model_name_from_file(filename):
    filename = filename.removesuffix(".json")
    return filename.split("-")[-1]

def get_test_from_file(filename, model_name):
    if model_name == "34b":
        test_list = filename.split("-")[:-1]
    else:
        test_list = filename.split("-")[:-2]
    temp = ""
    for test in test_list:
        temp += test + " "

    return temp

def write_confusion_matrix(confusion_matrix):
    TN = confusion_matrix["TN"]
    TP = confusion_matrix["TP"]
    FN = confusion_matrix["FN"]
    FP = confusion_matrix["FP"]
    
    interpreted_results.write("Confusion matrix for positive identification:\nTP: " + str(TP) + "    FN: "+ str(FN) + "\n")
    interpreted_results.write("FP: " + str(FP) + "    TN: " + str(TN) + "\n\n")

    return

def compute_F1_score(confusion_matrix):
    TN = confusion_matrix["TN"]
    TP = confusion_matrix["TP"]
    FN = confusion_matrix["FN"]
    FP = confusion_matrix["FP"]
    
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0

    if precision + recall != 0:
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

# Loading all files, only keep json files
og_dir_list = os.listdir()
dir_list = og_dir_list.copy()

for filename in og_dir_list:
    file_ext = filename.split(".")[-1]
    if "json" not in file_ext:
        dir_list.remove(filename)

        
        
# Treat all json files
for result_file in dir_list:
    model_name = get_model_name_from_file(result_file)
    test = get_test_from_file(result_file, model_name)
    gen_time = 0
    interpreted_results.write("------------------ Testing for " + model_name + " with " + test + "example ------------------\n\n")
    
    # If there is a flare
    TP = 0
    FN = 0

    # If there is no flare
    TN = 0
    FP = 0

    # If it did not give a 1-word yes/no answer and smth else was flagged as prediction
    not_respected = 0

    with open(result_file) as result_json:
        results = json.load(result_json)

    for image in results:
        if 'prediction' not in image.keys():
            continue
        image_name = image['image_dir'].split("/")[-1].removesuffix(".png")
        ground_truth = image['flare']
        predicted_class = image['prediction']
        gen_time += image['gen_time']

        if predicted_class != "no" and predicted_class != "yes":
            not_respected += 1

        if ground_truth == predicted_class:
            TP += 1 if ground_truth == "yes" else 0
            TN += 1 if ground_truth == "no" else 0

        else:
            FP += 1 if ground_truth == "no" else 0
            FN += 1 if ground_truth == "yes" else 0
# Raise flag is disabled because of the sheer number of test cases.            
#            raise_flag(image_name, predicted_class, ground_truth)

    interpreted_results.write("In the end, detected " + str(100 * TP/(TP + FN)) + "% of flares.\n")
    interpreted_results.write("In the end, detected " + str(100 * TN/(TN + FP)) + "% of no flares.\n")
    
    confusion_matrix = {"TP": TP, "FN": FN, "FP": FP, "TN": TN}
    write_confusion_matrix(confusion_matrix)
    write_F1_score(confusion_matrix)
    write_corr_coeff(np.asarray([confusion_matrix[key] for key in confusion_matrix]).reshape((2, 2)))
    interpreted_results.write("Did not follow instructions, answered with something else than yes/no and that was flagged as prediction: " + str(not_respected) + "\n")
    interpreted_results.write("Average answer time: " + str(gen_time/(TP + TN + FP + FN)) + "\n\n")

interpreted_results.close()
