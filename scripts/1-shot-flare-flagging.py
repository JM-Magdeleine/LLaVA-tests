"""
Script for testing 1-shot flare classification by LLaVA-NEXT on 100 labeled images.
Due to the dataset being already classified into two folders, positive and negative, the test is run on the whole dataset.
The images used for reference are:
    "036" for negative reference
    "0011" for obvious positive reference
    "0028" for hard-to-tell positive reference

NOTE: The only values to change between each exemple for testing are:
    The reference image path reference_image
    The prompt
    The result file name

Results are in a json file, with one file per model per test:
    image_dir: complete directory of tested image
    flare: ground truth on whether or not a flare is in the image
    prediction: the model's yes/no answer
    gen_time: time taken to generate the model's reponse
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

prompt = "This image is from the top of a train. If a flare appears, it is between the train's pantograph and catenary wires.  The first image contains a flare. Answering with either yes or no, is there a flare in the second image?"

test_files_dir = "/home/jmarie/flares/"
result_file_dir = "/home/jmarie/tests/results/1-shot-flare-flag/"
reference = os.path.join(os.path.join(test_files_dir, "positive_img"), "0028.png") 




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
    "tokenizer": tokenizer,
    "model": model,
    "image_processor": image_processor,
    "context_len": context_len
})()





# TESTING
image_dict_list = create_ground_truth_dict_list()
test_time = 0

for image in tqdm(image_dict_list):
  # Add exception that I might want to stop the test, but still record given answers cuz yea
  try:
    image_dir = image["image_dir"] + "," + reference
    args.image_file = image_dir
  
    prediction, gen_time = eval_model(args)
    prediction = sanitize_output(prediction)

    image["prediction"] = prediction
    image["gen_time"] = gen_time
  
    test_time += gen_time

  # If I interrupt, I still wanna record the results obtained thus far 
  except KeyboardInterrupt:
    break


with open(os.path.join(result_file_dir, "positive-difficult-" + model_string + ".json"), "w") as results_json:
  json.dump(image_dict_list, results_json)
  
print("Test time -----------------> ", test_time, "s")
