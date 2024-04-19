"""
Script for testing 1-shot flare color classification on 100 labeled images by LLaVA-NEXT.
Corresponding csv files in test-results:
"blue-obvious": The example is an obvious blue flare, 0492.png
"red-obvious": The example is an obvious red flare, 0836.png
"blue-hard": The example is a small blue flare, 0849.png
"red-hard": The example is a small red flare, 1299.png

Results are one csv file per model per example, with each column corresponding to:
    imgs: image files in the positive_img directory
    grund_truths: ground truth color red/blue
    predictions: predicted color red/blue
    gen_times: generation time for the given answer 
    difficulties: difficulty to predict image

NOTE: The only values that change from test to test are:
    reference: change to name of corresponding file
    output file: its name change correponding to which file it is tested on
"""
# IMPORTS
# Module imports for testing
import time, sys, re, json
from tqdm import tqdm
import pandas as pd

# Model imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_test import eval_model
# CHANGE _test TO _classification FOR ACTUAL CLASSIFICATION INSTEAD OF TEST







# WRAPPERS
def sanitize_output(output):
  # Parse output for flare color(red or blue)
  output = (output.lower()).strip()
  output = re.sub(r'[^\w\s]', '', output)
  output_words = output.split(" ")

  for word in output_words:
    if ("blue" in word) | ("red" in word):
      return word

  return ""

def get_img_file_dir(img):
  return test_files_dir + img + ".png"



# GLOBALS
# Command line
model_descriptor = sys.argv[1]

# LLaVA globals
if model_descriptor == 'light':
  model_path = "liuhaotian/llava-v1.6-vicuna-13b"
elif model_descriptor == 'full':
  model_path = "liuhaotian/llava-v1.6-34b"

prompt = "These two images are from the top of a train. In each image, there is a flare between the train's pantograph and the catenary wires. The first pitcure contains a blue flare. Is the flare in the second image red or blue?"

test_files_dir = "/home/jmarie/flares/positive_img/"
result_file_dir = "/home/jmarie/tests/results/1-shot-flare-color/"
with open('/home/jmarie/tests/test-files/classes-test.json') as img_json:
  imgs_dict = json.load(img_json)

reference = test_files_dir + "1299.png"


# LOADING MODEL
model_name = get_model_name_from_path(model_path)
model_string = model_name.replace("llava-v1.6-", "")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    load_4bit=True
)
args = type('Args', (), {
    "model_name": model_name,
    "query": prompt,
    "conv_mode": None,
    "image_file": "",
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 50,
  # Arguments import from eval_model to avoid loading the model at each test round
    "tokenizer": tokenizer,
    "model": model,
    "image_processor": image_processor,
    "context_len": context_len
})()





# TESTING
results_dict = {"imgs":[],"ground_truths":[],"predictions":[], "difficulties": [],"gen_times":[]}
test_time = 0

for img in tqdm(imgs_dict):
  img_path = get_img_file_dir(img['img'])
  image_file = reference + "," + img_path 
  flare_color = img['color']
  args.image_file = img_path

  if flare_color == "none":
    continue
  
  prediction, gen_time = eval_model(args)
  pred_color = sanitize_output(prediction)

  results_dict["imgs"].append(img['img'])
  results_dict["ground_truths"].append(img['color'])
  results_dict["predictions"].append(pred_color)
  results_dict["gen_times"].append(gen_time)
  results_dict["difficulties"].append(img['difficulty'])
  
  test_time += gen_time

results_df = pd.DataFrame(results_dict)
results_df.to_csv(result_file_dir + "red-hard-" + model_string + ".csv")

print("Test time -----------------> ", test_time, "s")
