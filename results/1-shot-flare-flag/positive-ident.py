import os, json

dir_list = os.listdir()
og_dir_list = dir_list.copy()

for file in og_dir_list:
    file_ext = file.split(".")[-1]

    if "json" not in file_ext or "vicuna" not in file:
        dir_list.remove(file)

print(dir_list)

output = {"image_files": []}

for results_file in dir_list:
    with open(results_file) as results_json:
        results = json.load(results_json)

    for image in results:
        if 'prediction' not in image.keys():
            continue

        if image['prediction'] == 'yes':
            image_dir = image['image_dir']
            if image_dir not in output['image_files']:
                output['image_files'].append(image['image_dir'])

with open("positive-results.json", 'w') as output_file:
    json.dump(output, output_file)
