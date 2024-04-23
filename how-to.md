# Structure
This is the testing directory. It consists of 3 main directories.
- test-files
- test-scripts
- test-results

## Test files
It contains the test files used. Naming standards will change, it only contains budget-test and its different resolutions for now. The cats-vs-dogs dataset was not included, but it can be found on Kaggle.

**Tests run on the model**:
- **Two-class classification**: cats vs. dogs
- **Spreadsheet comprehension**: budget test
  - _Gisting_: How well the model can understand what the different rows and columns correspond to
  - _Info retrieval_: How well the model has understood what it's looking at
  - _Data processing_: On top of understanding, can it think about what it's looking at?

## Test scripts
These are the scripts used for each test. Since there were only two, two scripts are present, one for each main test.

The scripts output the raw results in .csv format, recording the prompts.

### Cats vs. dogs:
- Prompt: "What animal do you see in this photo? Answer with either Cat or Dog."
- Results: table in the [pass/fail, file name] format.
The data processing is done with the test for this one.

### Spreadsheet:

- #### Gisting:
- Prompt: "What does the [nth] [column/row] do?"
- Results: table in the [recorded answer, test time] format
Maybe some modifications will be made to the prompt to get better answers.

- #### Information retrieval:

To be implemented.

- #### Data processing:

To be implemented

## Test results
This directory contains all test results in csv format and have a results processing script, to process the raw data stored for each test. The scripts currenty need to be implemented for each type of test.

### Cats vs. dogs
Tested on vicuna 13b. It does really well in terms of classification, but can sometimes be amiss for instruction following. Instead of answering with "dog", as asked, it sometimes will answer with puppy.

### Spreadsheet
The vicuna 13b and 34b model were tested. The first to be tested is a low-res image, up to HD image, with an intermediary med-res image. This is to see if there is a resolution, starting from which/up to which it works. There is then an adapted resolution, made so that it fits an integer number of patches for ViT.

- #### Gisting:
13b does poorly, oftentimes confusing columns for others, and rows for other columns. Hallucinates rows if not indicated whether to think about if the row is there or not. It consistently does those errors: It is reasonable to think that the resolution has little to no effect on error.

34b does much better, being very thorough and precise in the information it gives back.

- #### Information retrieval:
TBD

- #### Data processing:
TBD