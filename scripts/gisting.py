"""
Script for testing LLaVA-NEXT on its ability understand what it sees, by asking it to summarize lines and columns.
It includes testing for hallucination, by asking information about a column that does not exist.
"""
# IMPORTS
# Module imports for testing
import time, sys
import pandas as pd
import torch

# Model imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_test import eval_model




# WRAPPERS
def generate_prompt(line_or_column, location):
  # To generate prompt based on column and location
  return prompt_skeleton1 + location + line_or_column + pormpt_skeleton2



# GLOBALS
# Interpret command line inputs
model_descriptor = sys.argv[1]
image_descriptor = sys.argv[2]

# Test globals
outputs = []
gen_times = []

# LLaVA globals
locations = ["first", "second", "third", "fourth", "fifth"]
line_or_column = [" column", " line"]
prompt_skeleton1 = "What does the "
prompt_skeleton2 = " correspond to?"

test_files_dir = "/home/jmarie/tests/test-files/"
result_file_dir = "/home/jmarie/tests/results/spreadsheet/gisting/"
treated_descriptor = "-adapted-res" if image_descriptor == "adapted" else ("-high-res" if image_descriptor == "high" else ("-med-res" if image_descriptor == "med" else ""))
image_file_dir = test_files_dir + "budget-test" + treated_descriptor + ".png"



# MODEL LOADING
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

model_string = get_model_name_from_path(model_path).replace("llava-v1.6-", "")
args = type('Args', (), {
    "model_name": get_model_name_from_path(model_path),
    "query": "",
    "conv_mode": None,
    "image_file": image_file_dir,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 30,
  # Arguments import from eval_model to avoid loading the model with each try
    "tokenizer": tokenizer,
    "model": model,
    "image_processor": image_processor,
    "context_len": context_len
})()




# TESTING
start_time = time.time()
for line_column in line_or_column:
  for location in locations:
    prompt = generate_prompt(line_column, location)
    args.query = prompt
    output, gen_time = eval_model(args)
    outputs.append(output)
    gen_times.append(gen_time)
end_time = time.time()

print("Test time -----------------> ", end_time - start_time, "s")



# RESULTS EXPORT
outFrame = pd.DataFrame(data = [[output, gen_time] for output,gen_time in zip(outputs,gen_times)], columns = ["Outputs", "Generation time"])
outFrame.to_csv(result_file_dir + model_string + treated_descriptor + '.csv')
