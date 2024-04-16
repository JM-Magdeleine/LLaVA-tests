# Module imports for testing
import os, time
import plotext as plt
import numpy as np
import pandas as pd

# Model imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

def get_class_from_filename(filename: str):
  # Retrieve class corresponding to picture
  image_class = filename.split(".")[0]

  return image_class

def sanitize_class_name(class_name: str):
  # Format a given class, provided it is the first word output by LLaVA
  temp = class_name.split(" ")[0]
  temp.strip()

  return temp.lower()

def interpret_results(predictions, ground_truths, results):
  # Compares predictions to the ground truth on tested cases
  for i in range(len(predictions)):
    prediction = predictions[i]
    ground_truth = ground_truths[i]

    pred_was_right = prediction == ground_truth 

    # Did it predict a cat?
    if prediction == "cat":
      if pred_was_right:
        results[0] += 1

      else:
        # He got it wrong :(
        results[3] += 1

    # He predicted a dog.
    else:
      if pred_was_right:
        results[1] += 1
      else:
        # He got it wrong :(
        results[2] += 1
    
  return results

# Loading model
model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

# LLaVA globals
prompt = "What animal do you see in this photo? Answer with either Cat or Dog."
test_dir = "/home/jmarie/train/"
image_files = os.listdir(test_dir)
args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": "",
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 10
})()

# Test globals
num_tests = 10
results = [0, 0, 0, 0]
predictions, ground_truths = [], []
sample = []

# Test tim :)
i = 0
start_time = time.time()
random_sample = [np.random.randint(0, len(image_files)-1) for k in range(100)]
flags = []
random_image_files = [image_files[k] for k in random_sample]
out = []

for filename in random_image_files:
  i += 1
  print("test: ", i)
  ground_truth = get_class_from_filename(filename)

  # Prepare inputs for model prediction
  image_dir = test_dir + filename
  args.image_file = image_dir

  # Retrieve and format prediction
  pred_class = eval_model(args)
  pred_class = sanitize_class_name(pred_class)
  
  # Save results
  predictions.append(pred_class)
  ground_truths.append(ground_truth)
  if (pred_class != ground_truth):
    flags.append(filename)
    print(filename)
  out.append([pred_class, ground_truth, 1 if pred_class != ground_truth else 0])
  
end_time = time.time()
print("Test time: ", f"{end_time - start_time :.0f}", "s")

# Interpret results

interpreted_results = interpret_results(predictions, ground_truths, results)

print(predictions)
print(ground_truths)

plt.bar(["got cat right", "got dog right", "predicted dog was cat", "predicted cat was dog"], results)
plt.show()
print("Flags: ", flags)

outFrame = pd.DataFrame(data = out, columns = ["prediction", "ground truth", "flag"])
outFrame.to_csv('/home/jmarie/test-results/cat-dog.csv')
