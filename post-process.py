def get_input(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        t = ""
        flag = "skip"
        for line in f:
            if flag == "read":
                t += line
            if line.find("assistant") != -1:
                flag = "read"  
            elif not line.strip():
                flag = "skip"

    return t

def write_output(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(data)
        
def get_label(file_path):
    examples = []
    t = ""
    with open(file_path, "r", encoding="Shift-JIS") as f:
        for line in f:
            if not line.strip():
                examples.append(t)
                t = ""
            tokens = line.strip().split()
            if len(tokens) == 4:  # Ensure the correct format
                head_word, function_word, punctuation, tag = tokens
                t += (f"{head_word}\t{function_word}\t{punctuation}\t{tag}\n")
    return examples
        
def load_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        arr = []
        temp = []
        for line in f:
            if not line.strip():
               arr.append(temp)
               temp = []
            else:
                tokens = line.strip().split()
                temp.append(tokens)
    return arr 

def load_test_txt(filepath):
    with open(filepath, "r", encoding="Shift-JIS") as f:
        arr = []
        temp = []
        for line in f:
            if not line.strip():
               arr.append(temp)
               temp = []
            else:
                tokens = line.strip().split()
                temp.append(tokens)
    return arr 

def align_pred_label(pred_data, label_data):
    return 1

def write_align_output(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        for sentence in data:
            f.write(sentence + '\n')

    return 1

def process_pred():
    for test_id in range(1,11):
        label_test = f"japan-pension/test{test_id}.txt"
        label = get_label(label_test)
        filepath = f"output/Qwen/Qwen2.5-14B-Instruct/test{test_id}.txt"
        outpath = f"output/Qwen/Qwen2.5-14B-Instruct/test{test_id}-processed.txt"

        out = get_input(filepath)
        # print(out)
                            
        # print(label[0])
        write_output(outpath, out)


    # data = load_txt(outpath)
    # label_arr = load_test_txt(label_test)

    # print(len(data))
    # # print(data[0])
    # align_t = []
    # for i in range(len(data)):
    #     len_pred = len(data[i])
    #     len_label = len(label_arr[i])
    #     t = ""
    #     if len_pred >= len_label:
    #         for j in range(len_label):
    #             label_arr[i][j].append(data[i][j][-1])
    #             # print(label_arr[i][j])
    #             for m in range(5):
    #                 t += label_arr[i][j][m] + "\t"
    #             t += "\n"
                
    #     elif len_pred < len_label:
    #         for j in range(len_pred):
    #             label_arr[i][j].append(data[i][j][-1])
    #             # print(label_arr[i][j])
    #             for m in range(5):
    #                 t += label_arr[i][j][m] + "\t"
    #             t += "\n"
                
    #         for k in range(len_label-len_pred):
    #             label_arr[i][len_pred+k].append('O')
    #             # print(label_arr[i][len_pred+k])      
    #             for m in range(5):
    #                 t += label_arr[i][j][m] + "\t"
    #             t += "\n"
    #     print(t)
    #     align_t.append(t)
    #     # print(label_arr[i])

    # print(len(align_t))
    # write_align_output(f"output/Qwen/Qwen2.5-14B-Instruct/test{test_id}-align.txt", align_t)


if __name__ == "__main__":
    process_pred()
    


