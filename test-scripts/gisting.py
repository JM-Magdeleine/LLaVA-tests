# Module imports for testing
import time, sys
import pandas as pd

# Model imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_copy import eval_model

# Helper functions
def generate_prompt(line_or_column, location):
  return prompt_skeleton1 + location + line_or_column + prompt_skeleton2

# Loading model
if sys.argv[1] == 'light':
  model_path = "liuhaotian/llava-v1.6-vicuna-13b"
elif sys.argv[1] == 'full':
  model_path = "liuhaotian/llava-v1.6-34b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_4bit = True
)

# LLaVA globals
locations = ["first", "second", "third", "fourth", "fifth"]
line_or_column = [" column", " line"]
prompt_skeleton1 = "What does the "
prompt_skeleton2 = " correspond to?"
  # Choose between budget-test.png, budget-test-med-res.png, budget-test-high-res.png and budget-test-adapted-res.png
image_file_dir = "/home/jmarie/test-files/budget-test-adapted-res.png"

args = type('Args', (), {
  # Useless arguments that would be used to load the model in old eval_model func
  #  "model_path": model_path,
  #  "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": "",
    "conv_mode": None,
    "image_file": image_file_dir,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 30,
  # Arguments import from eval_model so that we don't have to load pretrained model each time
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
outFrame.to_csv('/home/jmarie/test-results/gisting-' + get_model_name_from_path(model_path) + '-adapted-res.csv')
