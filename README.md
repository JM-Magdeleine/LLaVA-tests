# Welcome
This is the repo for all tests run by me on LLAVA. It consists of 3 main directories:
- test-files
- scripts
- results

All tests, up to SNCF (cats vs.dogs, spreadsheet, SNCF), are run on 4-bit quantized Vicuna 13b and Hermes 34b. 

## Test files
It contains the test files used, minus the large datasets (namely flare/cadastres/cats-vs-dogs).

**Tests run on the model**:
- **Two-class classification**: cats vs. dogs
- **Spreadsheet comprehension**: budget test
  - _Summarizing_: Does the model make sense of what it's looking at?
  - _Gisting_: How well can the model understand what the different rows and columns correspond to?
  - _Restitution_: On top of understanding, can it think about what it's looking at?
- **More classification**: flares
  - _Zero-shot classification_: flare color and flare flagging, does it attend to small areas of large images, by itself?
  - _1-shot classification_: flare color and flare flagging, can it draw help from given examples, and do those examples orient its thinking in any way?

## Scripts
These are the scripts used for each test. At the beginning of each script is a summary of what each one does.

For budget test and cats vs. dogs classification, the scripts output the raw results in .csv format, recording the model's outputs.
From flares tests onwards, the output was in .json format. I found it easier to deal with for processing. Conventions have changed, but the latest is the one used for 1-shot flagging: {image directory, flag, predicted flag, generation time}.
_All files were named according to their respective tests_

### Cats vs. dogs:
- Prompt: "What animal do you see in this photo? Answer with either Cat or Dog."
- Results: table in the [pass/fail, file name] format.
The data processing is done with the test.

### Spreadsheet:
Results are output in a .csv format, the organization of which is specified in each script.

- #### Summarization:
- Prompt: "Describe each row and each column of this spreadsheet"

- #### Gisting:
- Prompt: "What does the [nth] [column/row] do?"

- #### Information retrieval:
- Prompt: "Convert this image into a .csv file:"

### Flare classification:
Results are output in a .json format, the organization of which is specified in each script. Slowly converging towards a stable and clear organization.

- #### Flare color:
- 0-shot: "This image is from the top of a train. There is a flare, between the train's pantograph and the catenary wires. In one word, is the flare in the image red or blue?"
- 1-shot: "These two images are from the top of a train. In each image, there is a flare between the train's pantograph and the catenary wires. The first pitcure contains a [blue/red] flare. Is the flare in the second image red or blue?"

- #### Flare flagging:
- 0-shot: "This image is from the top of a train. If a flare appears, it is between the train's pantograph and catenary wires. Answering with either yes or no, is there a flare in this image?"
- 1-shot: "These images are from the top of a train. If a flare appears, it is between the train's pantograph and catenary wires.  The first image [contains/does not contain] a flare. Answering with either yes or no, is there a flare in the second image?"
  - **This test is also run on un-quantized 7b models (Mistral and Vicuna)**

_Note_: There are 2 variants of flare color 0-shot. One is for the test on 100 hand-labeled images (-test), while the other is to classify the whole dataset (-classification). There are also 2 variants of flare flagging 1-shot. One is for manually testing each example variation, while the other is automated to run all example variations (-automated).

## Results
This directory contains all test results in csv format and have a results processing script, to process the raw data stored for each test. The scripts currenty need to be implemented for each type of test.
_Note: The interpretation of results is not complete yet_

### Cats vs. dogs
Tested on vicuna 13b. It does really well in terms of classification, but can sometimes be amiss for instruction following. Instead of answering with "dog", as asked, it sometimes will answer with puppy.

### Spreadsheet
The vicuna 13b and 34b model were tested. The first to be tested is a low-res image, up to HD image, with an intermediary med-res image. This is to see if there is a resolution, starting from which/up to which it works. There is then an adapted resolution, made so that it fits an integer number of patches for ViT.

- #### Summarization:



- #### Gisting:
13b does poorly, oftentimes confusing columns for others, and rows for other columns. Hallucinates rows if not indicated whether to think about if the row is there or not. It consistently does those errors: It is reasonable to think that the resolution has little to no effect on error.

34b does much better, being very thorough and precise in the information it gives back.

- #### :
TBD

- #### Data processing:
TBD