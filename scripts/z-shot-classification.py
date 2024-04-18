"""
Script for zero-shot classification of positive flare images, distinguishing between red and blue flares.

Based on the results on 100 images, we can be confident in Vicuna's blue answers. The same can be said about Hermes's red answers.

Thus, I first classify the for sure blues with Vicuna, putting them in a dedicated array. Then, I classify the "for sure" red with Hermes, putting them in a dedicated array.
If there are any remaining images not classfied as blue by Vicuna or red by Hermes, they will be classified with Vicuna. This model was decided, by using r value for correlation to diagonal matrix (see https://math.stackexchange.com/questions/1392491/measure-of-how-much-diagonal-a-matrix-is for more info), because Vicuna is best at classifying for color, **on average**

Because of memory constraints, model loading has been refactored to only be contained in the heap when classifying, in order to free memory at each model loading.
"""

# IMPORTS
import os, re, json, time, torch, gc
from tqdm import tqdm

# Model imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava_flares import eval_model

# WRAPPERS
def get_color_from_output(output):
  # Parse output string from model for "red" or "blue" 
  output = (output.lower()).strip()
  output = re.sub(r'[^\w\s]', '', output)
  # Above pattern pulled from https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
  output_words = output.split(" ")

  for word in output_words:
    if ("blue" in word) | ("red" in word):
      return word

  return ""

def classify(model_path, img_list, desc):
  # Main classifying function
  img_to_classify = img_list.copy()
  model_name = get_model_name_from_path(model_path)
  
  # Load model and call parameters
  tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    load_4bit=True
  )
  args = type('Args', (), {
    "model_name": model_name,
    "query": "This image is from the top of a train. There is a flare, between the train's pantograph and the catenary wires. In one word, is the flare in the image red or blue?",
    "conv_mode": None,
    "image_file": "",
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 100,
    "tokenizer": tokenizer,
    "model": model,
    "image_processor": image_processor,
    "context_len": context_len
  })()

  # Run classification
  for img in tqdm(img_list, desc=desc):
    args.image_file = files_dir + img
    
    prediction = eval_model(args)
    color = get_color_from_output(prediction)

    img_dict = {"image_file": img, "flare_color": color}
    if (color in desc.lower()) or ("remaining" in desc.lower()):
      img_to_classify.remove(img)
      img_dict_list.append(img_dict)

  del model, tokenizer, image_processor, context_len, args, model_name

  return img_to_classify




# VARIABLES
# Globals
files_dir = "/home/jmarie/flares/positive_img/"
result_dir = "/home/jmarie/flares/"
img_dict_list = []
img_list = os.listdir(files_dir)

# Model variables
vicuna_path = "liuhaotian/llava-v1.6-vicuna-13b"
hermes_path = "liuhaotian/llava-v1.6-34b"




# CLASSIFICATION
# Run blue classification with Vicuna
img_list = classify(vicuna_path, img_list, "Vicuna BLUE")
gc.collect()
torch.cuda.empty_cache()

# Run red classification with Hermes
img_list = classify(hermes_path, img_list, "Hermes RED")
gc.collect()
torch.cuda.empty_cache()

# Run remaining classification with Vicuna
if len(img_list) != 0:
  img_list = classify(vicuna_path, img_list, "Vicuna REMAINING")
  gc.collect()
  torch.cuda.empty_cache()


# OUTPUTTING TO JSON
with open(result_dir + '/colors.json', 'w') as colors:
  json.dump(img_dict_list, colors)
