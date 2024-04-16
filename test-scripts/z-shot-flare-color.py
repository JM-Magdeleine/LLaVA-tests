# Module imports for testing
import time, sys, re, json
import pandas as pd

# Model imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_test import eval_model #run_llava was modified so as to not reload the model at each eval_model call, USE _test VERSION

# Helper functions
def sanitize_output(output):
  output = (output.lower()).strip()
  output = re.sub(r'[^\w\s]', '', output)
  output_words = output.split(" ")

  for word in output_words:
    if ("blue" in word) | ("red" in word):
      return word

  return ""

def get_img_file_dir(img):
  return test_files_dir + img + ".png"

# Interpret command line inputs
model_descriptor = sys.argv[1]

# Loading model
if model_descriptor == 'light':
  model_path = "liuhaotian/llava-v1.6-vicuna-13b"
elif model_descriptor == 'full':
  model_path = "liuhaotian/llava-v1.6-34b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_4bit=True
)

# LLaVA globals
prompt = "This image is from the top of a train. There is a flare, between the train's pantograph and the catenary wires. In one word, is the flare in the image red or blue?"

test_files_dir = "/home/jmarie/flares/positive_img/"
result_file_dir = "/home/jmarie/tests/test-results/z-shot-flare-color/"
with open('/home/jmarie/flares/classes-test.json') as img_json:
  imgs_dict = json.load(img_json)

model_string = get_model_name_from_path(model_path).replace("llava-v1.6-", "")
args = type('Args', (), {
    "model_name": get_model_name_from_path(model_path),
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

# Run test
results_dict = {"imgs":[],"ground_truths":[],"predictions":[], "difficulties": [],"gen_times":[]}
test_time = 0

for img in imgs_dict:
  img_path = get_img_file_dir(img['img'])
  flare_color = img['color']
  args.image_file = img_path

  if flare_color == "none":
    continue
  
  prediction, gen_time = eval_model(args)
  sanitized_output = sanitize_output(prediction)

  results_dict["imgs"].append(img['img'])
  results_dict["ground_truths"].append(img['color'])
  results_dict["predictions"].append(sanitized_output)
  results_dict["gen_times"].append(gen_time)
  results_dict["difficulties"].append(img['difficulty'])
  
  test_time += gen_time

results_df = pd.DataFrame(results_dict)
results_df.to_csv(result_file_dir + "100-imgs-" + (args.model_name).replace("llava-v1.6-","") + ".csv")

print("Test time -----------------> ", test_time, "s")
