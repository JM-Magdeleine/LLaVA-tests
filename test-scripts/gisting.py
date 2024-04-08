# Module imports for testing
import time, sys
import pandas as pd
import torch

# Model imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_test import eval_model

# Interpret command line inputs
model_descriptor = sys.argv[1]
image_descriptor = sys.argv[2]

# Helper functions
def generate_prompt(line_or_column, location):
  return prompt_skeleton1 + location + line_or_column + prompt_skeleton2

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
locations = ["first", "second", "third", "fourth", "fifth"]
line_or_column = [" column", " line"]
prompt_skeleton1 = "What does the "
prompt_skeleton2 = " correspond to?"

treated_descriptor = "-adapted-res" if image_descriptor == "adapted" else ("-high-res" if image_descriptor == "high" else ("-med-res" if image_descriptor == "med" else ""))
image_file_dir = "/home/jmarie/test-files/budget-test" + treated_descriptor + ".png"

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

# Test variables
outputs = []
gen_times = []

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
outFrame = pd.DataFrame(data = [[output, gen_time] for output,gen_time in zip(outputs,gen_times)], columns = ["Outputs", "Generation time"])
outFrame.to_csv('/home/jmarie/test-results/gisting/gisting-' + model_string + treated_descriptor + '.csv')
