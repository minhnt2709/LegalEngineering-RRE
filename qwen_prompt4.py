import os
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the LLaMA-3 model and tokenizer
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# access_token = "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"

template = """You are a legal expert in Japan Pension Law. Perform the sequence labeling task based on logical structure of the sentence.
- There are two levels of structure of a sentence: Logical Part and Logical Structure. 
- Logical Part contains antecedence parts (A), consequence parts (C), and topic parts (T). A consequent part describes a law provision, an antecedent part describes cases or the context in which the law provision can be applied, and a topic part describes subjects related to the law provision. 
- Logical Structure is divided into Requisite part (R) and Effectuation part (E). In more detail, requisite and effectuation parts are constructed from one or more logical parts such as antecedence parts, consequence parts, and topic parts.
- Based on above knowledge, there are four types of legal sentence, including: 
- Case 0: the requisite part (R) only consists of one antecedent part (A) and the effectuation part (E) only consists of one consequent part (C)
- Case 1:  the requisite part (R) consists of one antecedent part (A) and one topic part (T1) that depends on the antecedent part (A).
- Case 2:  the effectuation part (E) consists of one consequent part (C) and one topic part (T2) that depends on the consequent part (C).
- Case 3: both requisite part (R) and effectuation part (E) contains topic part (T3).

Tags to use: 
- E: Consequent parts in case 1, 2, 3
- EL: Requisite part in case 0
- ER: Effectuation part in case 0
- R: Antecedent parts in case 1, 2, 3
- S1: Topic part in case 1
- S2: Topic part in case 2
- S3: Topic part in case 3
- I: Inside (e.g, I-R, I-E)
- E: Ending (e.g, E-R, E-E, E-S1, E-S2, E-S3), note that E means end of a chunk
- O: outside

Let's think step by step. 
Step 1: Count number of input tokens in the sequence, indicated as N.
Step 2: Analyze the logical parts in the sequence.
Step 3: Analyze the logical structures in the sequence.
Step 4: Label the token based on analyses.
Constrain: Only generate tags table, do not generate explanations. The number of output tags must be equal to N. 

Example 1:
Head word | Function word | Punctuation | Tag
事業	の	NO	I-S2
事務	の	NO	I-S2
一部	は	、	E-S2
政令	の	NO	I-E
定める	定める	NO	I-E
ところ	に	NO	I-E
より	より	、	I-E
法律	によって	NO	I-E
さ	た	NO	I-E
組合	組合	、	I-E
会	会	、	I-E
会	会	、	I-E
法	の	NO	I-E
規定	により	NO	I-E
制度	を	NO	I-E
する	する	NO	I-E
こと	と	NO	I-E
さ	た	NO	I-E
団	に	NO	I-E
行わ	せる	NO	I-E
こと	が	NO	I-E
できる	できる	。	E-E

Example 2:
Head word | Function word | Punctuation | Tag
政府	は	、	E-S3
項	の	NO	I-R
規定	により	NO	I-R
財政	の	NO	I-R
及び	及び	NO	I-R
見通し	を	NO	I-R
し	た	NO	I-R
とき	は	、	E-R
遅滞	遅滞	NO	I-E
なく	なく	、	I-E
これ	を	NO	I-E
し	ない	。	E-E

Example 3:
Head word | Function word | Punctuation | Tag
この	この	NO	I-EL
法律	において	、	I-EL
法	は	、	E-EL
次	の	NO	I-ER
号	に	NO	I-ER
掲げる	掲げる	NO	I-ER
法律	を	NO	I-ER
いう	いう	。	E-ER

Example 4:
Head word | Function word | Punctuation | Tag
この	この	NO	O
法律	において	、	O
者	者	、	I-S2
及び	及び	NO	I-S2
妻	は	、	E-S2
婚姻	の	NO	I-E
届出	を	NO	I-E
し	が	、	I-E
上	上	NO	I-E
関係	と	NO	I-E
同様	の	NO	I-E
事情	に	NO	I-E
ある	ある	NO	I-E
者	を	NO	I-E
含む	含む	NO	I-E
もの	と	NO	I-E
する	する	。	E-E

Example 5:
Head word | Function word | Punctuation | Tag
この	この	NO	I-EL
法律	において	、	I-EL
期間	は	、	E-EL
項	項	NO	I-ER
号	に	NO	I-ER
する	する	NO	I-ER
者	の	NO	I-ER
期間	の	NO	I-ER
うち	うち	NO	I-ER
さ	た	NO	I-ER
料	に	NO	I-ER
係る	係る	NO	I-ER
もの	もの	、	I-ER
項	項	NO	I-ER
号	に	NO	I-ER
する	する	NO	I-ER
者	の	NO	I-ER
及び	及び	NO	I-ER
項	項	NO	I-ER
号	に	NO	I-ER
する	する	NO	I-ER
者	の	NO	I-ER
期間	を	NO	I-ER
し	た	NO	I-ER
期間	を	NO	I-ER
いう	いう	。	E-ER

Example 6:
Head word | Function word | Punctuation | Tag
し	た	NO	I-S2
基金	は	、	E-S2
規約	の	NO	I-E
定める	定める	NO	I-E
ところ	に	NO	I-E
より	より	、	I-E
項	の	NO	I-E
規定	により	NO	I-E
員	に	NO	I-E
す	べき	NO	I-E
財産	の	NO	I-E
交付	を	NO	I-E
項	の	NO	I-E
規定	により	NO	I-E
金	に	NO	I-E
する	する	NO	I-E
額	を	NO	I-E
し	た	NO	I-E
会	に	NO	I-E
申し出る	申し出る	NO	I-E
こと	が	NO	I-E
できる	できる	。	E-E

Example 7:
Head word | Function word | Punctuation | Tag
事業	は	、	E-S2
政府	が	、	I-E
する	する	。	E-E

Example 8:
Head word | Function word | Punctuation | Tag
この	この	NO	I-EL
法律	において	、	I-EL
期間	は	、	E-EL
項	項	NO	I-ER
号	に	NO	I-ER
する	する	NO	I-ER
者	の	NO	I-ER
期間	の	NO	I-ER
うち	うち	NO	I-ER
さ	た	NO	I-ER
料	に	NO	I-ER
係る	係る	NO	I-ER
もの	もの	、	I-ER
項	項	NO	I-ER
号	に	NO	I-ER
する	する	NO	I-ER
者	の	NO	I-ER
及び	及び	NO	I-ER
項	項	NO	I-ER
号	に	NO	I-ER
する	する	NO	I-ER
者	の	NO	I-ER
期間	を	NO	I-ER
し	た	NO	I-ER
期間	を	NO	I-ER
いう	いう	。	E-ER

Example 9: 
Head word | Function word | Punctuation | Tag
長官	は	、	E-S2
者	から	、	I-R
又は	又は	NO	I-R
貯金	の	NO	I-R
払出し	と	NO	I-R
その	その	NO	I-R
払い出し	た	NO	I-R
金銭	による	NO	I-R
料	の	NO	I-R
納付	を	NO	I-R
その	その	NO	I-R
又は	又は	NO	I-R
口座	の	NO	I-R
ある	ある	NO	I-R
機関	に	NO	I-R
し	て	NO	I-R
行う	行う	NO	I-R
こと	を	NO	I-R
する	する	NO	I-R
旨	の	NO	I-R
申出	が	NO	I-R
あっ	た	NO	I-R
場合	は	、	E-R
その	その	NO	I-E
納付	が	NO	I-E
確実	と	NO	I-E
認め	られ	、	I-E
かつ	かつ	、	I-E
その	その	NO	I-E
申出	を	NO	I-E
する	する	NO	I-E
こと	が	NO	I-E
料	の	NO	I-E
上	上	NO	I-E
有利	と	NO	I-E
認め	られる	NO	I-E
とき	に	NO	I-E
限り	限り	、	I-E
その	その	NO	I-E
申出	を	NO	I-E
する	する	NO	I-E
こと	が	NO	I-E
できる	できる	。	E-E

Example 10:
Head word | Function word | Punctuation | Tag
者	が	NO	I-R
項	の	NO	I-R
規定	により	NO	I-R
料	を	NO	I-R
者	に	NO	I-R
し	た	NO	I-R
とき	は	、	E-R
当該	当該	NO	I-S2
料	に	NO	I-S2
係る	係る	NO	I-S2
期間	は	、	E-S2
条	条	NO	I-E
項	の	NO	I-E
規定	の	NO	I-E
適用	は	NO	I-E
期間	と	NO	I-E
みなす	みなす	。	E-E
"""

def prompt_construction(template, input):
    return template.replace('[[INPUT]]', input)

def load_input_file(file_path):
    examples = []
    t = ""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                examples.append(t)
                t = ""
            tokens = line.strip().split()
            if len(tokens) == 3:  # Ensure the correct format
                head_word, function_word, punctuation = tokens
                t += (f"{head_word}\t{function_word}\t{punctuation}\n")
    return examples

def write_output(file_path, output):
    with open(file_path, "w", encoding="utf-8") as f:
        for out in output:
            f.write(out + "\n\n")


# Function to perform NER using the LLaMA-3 model
def perform_ner(model, tokenizer, template, input):
    # Format the prompt with examples and the input sentence
    prompt = prompt_construction(template, input)
    # print(prompt)
    s = "Head word | Function word | Punctuation | Tag"
    input = s + "\n" + input
    messages = [
        {"role": "system", "content": template},
        {"role": "user", "content": input}
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
            max_new_tokens=800,  # Limit output length
            do_sample=True
        )
    
    # Decode and return the output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.split("Output:\n")[-1].strip()

def write_output(file_path, output):
    with open(file_path, "w", encoding="utf-8") as f:
        for out in output:
            f.write(out + "\n\n")


def main():
    test_files = ["test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9", "test10"]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda", torch_dtype=torch.float16)
    model.generation_config.cache_implementation = "static"
    model.to(DEVICE)
    print("Model loaded successfully.")

    directory = f"output/prompt4/{MODEL_NAME}"
    if not os.path.exists(directory):
        os.makedirs(directory)
            
    for test in test_files:
        print(test)
        test_file = f"japan-pension-test/{test}.txt"
        test_data = load_input_file(test_file)

        output_file = f"output/prompt4/{MODEL_NAME}/{test}.txt"
        print(len(test_data))
        
        arr = []
        for i in tqdm(range(len(test_data))):
            # t = prompt_construction(template, test_data[i])
            # print(t)
            res = perform_ner(model, tokenizer, template, test_data[i])
            # print(res)
            arr.append(res)
            write_output(output_file, arr)
        
        break

if __name__ == "__main__":
    main()
    
