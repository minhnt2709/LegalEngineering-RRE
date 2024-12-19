import os
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the LLaMA-3 model and tokenizer
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# access_token = "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"
prompt_opt = 2

# Function to load and parse the input text file
def load_input_file(file_path, mode):
    examples = []
    t = ""
    with open(file_path, "r", encoding="Shift-JIS") as f:
        for line in f:
            if not line.strip():
                examples.append(t)
                t = ""
            tokens = line.strip().split()
            if mode == 'train':
                if len(tokens) == 4:  # Ensure the correct format
                    head_word, function_word, punctuation, tag = tokens
                    t += (f"{head_word}\t{function_word}\t{punctuation}\t{tag}\n")
            elif mode == 'test':
                if len(tokens) == 4:  # Ensure the correct format
                    head_word, function_word, punctuation, tag = tokens
                    t += (f"{head_word}\t{function_word}\t{punctuation}\n")
    return examples

def load_json_file(file_path):
    r = []
    with open(file_path, "r", encoding="Shift-JIS") as f:
        data = json.load(f)
        
    for sentence in data:
        head = []
        function = []
        punc = []
        tag = []
        for line in sentence: 
            head.append(line["Head word"])
            function.append(line["Function word"])
            punc.append(line["Punctuation"])
            tag.append(line["Tag"])
        r.append({"Head word": head, "Function word": function, "Punctuation": punc, "Tag": tag})
    return r

# Function to format examples for few-shot learning
def format_prompt(examples, input_sentence, opt):
    """
    Prompt1: Table 
    Prompt2: Dict 
    """
    if opt == 1:
        prompt = "Perform Named Entity Recognition (NER) on Japanese National Pention Law with the following tags:\n"
        prompt += "S - Subject, E - Effectuation, R - Requisite; O for outside; I for inside (e.g., I-E) and E for ending (e.g., E-E). Do not generate explaination.\n\n"
        prompt += "Examples:\n"
        prompt += "Head word\tFunction word\tPunctuation\tTag\n"
        prompt += examples
        prompt += "\nInput:\n" + input_sentence + "\nOutput:\n"
    elif opt == 2:
        prompt = "Perform Named Entity Recognition (NER) on Japanese National Pention Law with the following tags:\n"
        prompt += "S - Subject, E - Effectuation, R - Requisite; O for outside; I for inside (e.g., I-E) and E for ending (e.g., E-E). You must generate only NER output, do not generate anything else.\n\n"
        prompt += "Examples:\n"
        prompt += "\nInput:\n" + "Head word: " + str(examples["Head word"]) + "\n" + "Function word" + str(
            examples["Function word"]) + "\n" + "Punctuation: " + str(examples["Punctuation"]) + "\n"
        prompt += "\nOutput:\n" + "Tag: " + str(examples["Tag"]) + "\n"
        prompt += "\nInput:\n" + "Head word: " + str(input_sentence["Head word"]) + "\n" + "Function word: " + str(
            input_sentence["Function word"]) + "\n" + "Punctuation: " + str(input_sentence["Punctuation"]) + "\nOutput:\n"
    return prompt


# Function to perform NER using the LLaMA-3 model
def perform_ner(model, tokenizer, examples, input_sentence, opt):
    """
    Generates NER tags for the input sentence using the LLaMA-3 model.
    """
    # Format the prompt with examples and the input sentence
    prompt = format_prompt(examples, input_sentence, opt)
    # print(prompt)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # Tokenize the prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    # Generate output
    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            max_length=model_inputs['input_ids'].shape[1] + 100,  # Limit output length
            do_sample=False  # Greedy decoding
        )
    
    # Decode and return the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.split("Output:\n")[-1].strip()

def export_results(output_file, res):
    with open(output_file, "w", encoding="utf-8") as f:
        for i in res:
            f.write("\nNER output:\n" + "Head word: " + str(i["Head word"]) + "\n" + "Function word: " + str(i["Function word"]) + 
                "\n" + "Punctuation: " + str(i["Punctuation"]) + "\n" + "Tag: " + str(i["Tag"]) + 
                "\n" + "Output" + str(i["Output"]) + "\n\n")

# Main function
def main():
    # Prompt 1
    # test_file = "japan-pension/test3.txt"
    # train_file = "japan-pension/train2.txt"
    # examples = load_input_file(train_file, "train")
    # test_samples = load_input_file(test_file, "test")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
    # model.to(DEVICE)
    print("Model loaded successfully.")
    
    # Prompt 2
    test_name = ["test4", "test5", "test6", "test7", "test8", "test9", "test10"]
    
    for file in test_name:
        print(file)
        test_file = f"japan-pension-js/{file}.json"
        train_file = "japan-pension-js/train2.json"
        examples = load_json_file(train_file)
        test_samples = load_json_file(test_file)
        

        # Perform NER
        res = []
        for i in tqdm(range(len(test_samples))):
            print("Performing NER...")
            # print(test_samples[i])
            output = perform_ner(model, tokenizer, examples[5], test_samples[i], opt=prompt_opt)
            test_samples[i]["Output"] = output
            res.append(test_samples[i])
            print(res)
        
        directory = f"output/prompt{prompt_opt}/{MODEL_NAME}"
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        output_file = f"output/prompt{prompt_opt}/{MODEL_NAME}/{file}.txt"
        export_results(output_file, res)

if __name__ == "__main__":
    main()
    
