# Module imports for testing
import time, sys
import pandas as pd
import torch
from pytimedinput import timedInput

# Model imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_test import eval_model #run_llava was modified so as to not reload the model at each eval_model call, USE _test VERSION

# Helper functions
def OutputToDF(output):
  # Function to take output of the LLM and convert it to a pd DataFrame
  
  # Not elegant way to ignore newline characters, couldn't get them to strip with split with re
  csv_txt = output.split("```")[1][1:-1]

  csv_lines = csv_txt.split("\n")

  csv_format = [csv_line.split(",") for csv_line in csv_lines]

  return pd.DataFrame(data = csv_format[1:], columns =csv_format[0])

def get_image_file_dir(image_descriptor):
  treated_descriptor = "-adapted-res" if image_descriptor == "adapted" else ("-high-res" if image_descriptor == "high" else ("-med-res" if image_descriptor == "med" else ""))
  image_file_dir = test_files_dir + "budget-test" + treated_descriptor + ".png"

  return image_file_dir, treated_descriptor

# Interpret command line inputs
model_descriptor = sys.argv[1]
image_descriptor = sys.argv[2]

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
prompt = "Convert this image into a .csv file:"

test_files_dir = "/home/jmarie/tests/test-files/"
result_file_dir = "/home/jmarie/tests/test-results/restitution/"
image_file_dir, treated_descriptor = get_image_file_dir(image_descriptor)

model_string = get_model_name_from_path(model_path).replace("llava-v1.6-", "")
args = type('Args', (), {
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file_dir,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 500,
  # Arguments import from eval_model to avoid loading the model at each test round
    "tokenizer": tokenizer,
    "model": model,
    "image_processor": image_processor,
    "context_len": context_len
})()

# Run test
run_test = True
test_time = 0

while run_test:
  # Run the test
  output, gen_time = eval_model(args)
  outFrame = OutputToDF(output)
  outFrame.to_csv(result_file_dir + model_string + treated_descriptor + '.csv')
  test_time += gen_time
  
  # Once again?
  run_test, timed_out = timedInput(prompt = "Model has finished generating, continue the test ?\n",
                        timeout = 10)
  run_test = True if run_test == 'y' else False

  if run_test:
    # If another test, change variable for new prompt context
    image_file_dir, treated_descriptor = get_image_file_dir(input("Which image to test next?\n"))
    args.image_file = image_file_dir


print("Test time -----------------> ", test_time, "s")
