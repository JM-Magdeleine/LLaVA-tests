"""
Script for testing 0-shot flare classification by LLaVA-NEXT on 100 labeled images.
Due to the dataset being already classified into two folders, positive and negative, the test is run on the whole dataset.

RESULTS are output as json file for each model tested, as a list of dicts:
    image_dir: image's absolute path
    flare: flag whether there is a flare or not
    prediction: 
    gen_time:
"""
# IMPORTS
# Module imports for testing
import time, sys, re, json, os
from tqdm import tqdm

# Model imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_test import eval_model







# WRAPPERS
def create_ground_truth_dict_list():
  # Returns the list of ground truth dictionaries, containing the absolute path of the pictures, and their corresponding flare flag. During testing, predicted values and gen times will be added to those dictionaries
  image_dict_list = []

  no_flare = os.listdir(os.path.join(test_files_dir, "negative_img"))
  yes_flare = os.listdir(os.path.join(test_files_dir, "positive_img"))
  for image in no_flare:
    image_dir = os.path.join(os.path.join(test_files_dir, "negative_img"), image)
    image_dict = {"image_dir": image_dir, "flare": "no"}
    image_dict_list.append(image_dict)

  for image in yes_flare:
    image_dir = os.path.join(os.path.join(test_files_dir, "positive_img"), image)
    image_dict = {"image_dir": image_dir, "flare": "yes"}
    image_dict_list.append(image_dict)

  return image_dict_list

def sanitize_output(output):
  # Parse output for flare (yes or no)
  output = (output.lower()).strip()
  output = re.sub(r'[^\w\s]', ' ', output)
  output_words = output.split(" ")

  for word in output_words:
    if ("no" in word) | ("yes" in word):
      return word

  return ""


# GLOBALS
# Command line arguments
model_descriptor = sys.argv[1]

# LLaVA globals
if model_descriptor == 'light':
  model_path = "liuhaotian/llava-v1.6-vicuna-13b"
elif model_descriptor == 'full':
  model_path = "liuhaotian/llava-v1.6-34b"

prompt = "This image is from the top of a train. If a flare appears, it is between the train's pantograph and catenary wires. Answering with either yes or no, is there a flare in this image?"

test_files_dir = "/home/jmarie/flares/"
result_file_dir = "/home/jmarie/tests/results/1-shot-flare-flag/"





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
image_dict_list = create_ground_truth_dict_list()
test_time = 0

for image in tqdm(image_dict_list):
  image_dir = image["image_dir"]
  args.image_file = image_dir
  
  prediction, gen_time = eval_model(args)
  prediction = sanitize_output(prediction)

  image["prediction"] = prediction
  image["gen_time"] = gen_time
  
  test_time += gen_time

with open(os.path.join(result_file_dir, model_string + ".json"), "w") as results_json:
  json.dump(image_dict_list, results_json)
  
print("Test time -----------------> ", test_time, "s")
