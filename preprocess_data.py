import json

# Read a text file containing Japanese words and print its content
def read_japanese_file(file_path):
    # Open the file with UTF-8 encoding to handle Japanese text
    data = []
    r = []
    with open(file_path, 'r', encoding="Shift-JIS") as file:
        for line in file:
            if len(line) <= 1:
                r.append(data)   
                data = []
            data.append(line.strip())
            
    return r        

def convert_to_json(input_file, output_file):
    # Initialize an empty list to hold the dictionary data
    data = []
    r = []
    # Open the input file and process each line
    with open(input_file, 'r', encoding='Shift-JIS') as file:
        for line in file:
            # Skip empty lines
            if not line.strip():
                r.append(data)
                data = []
            if line.strip():
                # Split the line into columns
                columns = line.strip().split()
                # Ensure the line has the correct format
                if len(columns) == 4:
                    head_word, function_word, punctuation, tag = columns
                    # Append a dictionary for each line
                    data.append({
                        "Head word": head_word,
                        "Function word": function_word,
                        "Punctuation": punctuation,
                        "Tag": tag
                    })

    # Write the data to a JSON file
    with open(output_file, 'w', encoding='Shift-JIS') as json_file:
        json.dump(r, json_file, ensure_ascii=False, indent=4)

def del_label(input_file, output_file):
    with open(input_file, "r", encoding="Shift-JIS") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                outfile.write('\n')
            else:
                col = line.strip().split()
                t = col[:-1]
                outfile.write("\t".join(t) + "\n")
                

# Input and output file paths
for i in range(1, 11): 
    input_file = f"japan-pension/test{i}.txt"  # Replace with your input file name
    output_file = f"japan-pension-test/test{i}.txt"  # Replace with your desired output file name
    del_label(input_file, output_file)

