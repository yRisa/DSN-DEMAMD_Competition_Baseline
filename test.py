import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time


def ask(question,history=[],url="http://127.0.0.1:8012/ask"):
    data = {'question': question, 'history': history}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def process_file(file_index, item_name, prompt1, input_dir,output_dir,max_thread):
    if file_index<max_thread:
        #在函数中记录自己的线程号
        threading.current_thread().name = f"{file_index}"
        cur_thread=int(threading.current_thread().name)
        time.sleep(1)
    else:
        cur_thread = int(threading.current_thread().name)

    url="http://127.0.0.1:8012/ask"

    file_path = os.path.join(input_dir, item_name)
    output_path = os.path.join(output_dir, item_name)
    if os.path.exists(output_path):
        return f"File '{output_path}' already exists. Skipping."

    print(f"Processing file {item_name}...")
    with open(file_path, 'r', encoding='utf-8') as file:
        cur_txt = file.read()
        prompt = cur_txt
        result = ask(prompt,url=url)
        if result is None:
            pass
        else:
            json.dump(result, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


def main(directory,tmp_directory1,tmp_directory2,output_directory, n_threads):

    if not os.path.exists(tmp_directory1):
        os.makedirs(tmp_directory1)
    if not os.path.exists(tmp_directory2):
        os.makedirs(tmp_directory2)
    
    #请修改prompt
    prompt='请抽取因果关系事件对'
    tmp_list=[]

    with open(directory, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        #print(item)
        q=item["text"]+prompt
        a=json.dumps(item["causality_list"],ensure_ascii=False,indent=4)
        no=item["document_id"]
        tmp_list.append([q,a,no])
        with open(os.path.join(tmp_directory1,str(no)),'w',encoding='utf-8') as f:
            f.write(q)

    filenames = [f for f in os.listdir(tmp_directory1) if os.path.isfile(os.path.join(tmp_directory1, f))]
    print(filenames)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_file = {executor.submit(process_file, i, fname, prompt, tmp_directory1, tmp_directory2, n_threads): fname for i, fname in
                          enumerate(filenames)}
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                result = future.result()
                print(f"Completed processing {file_name}: {result}")
            except Exception as exc:
                print(f"Error processing {file_name}: {str(exc)}")

    output=[]
    filenames2 = [f for f in os.listdir(tmp_directory2) if os.path.isfile(os.path.join(tmp_directory2, f))]
    for item in filenames2:
        try:
            with open(os.path.join(tmp_directory2,item),'r',encoding='utf-8') as f:
                data=json.load(f)
                data2=data["answer"]

                data2=data2.replace("'",'"')
                start = data2.find("[")
                end = data2.rfind("]")
                data = data2[start:end + 1]
                data = json.loads(data)

                to_pop=[]
                for i,item2 in enumerate(data):
                    if 'cause' not in data[i].keys() or 'effect' not in data[i].keys():
                        to_pop.append(i)
                for item in to_pop:
                    data.pop(item)

            output.append({"document_id":int(item),"causality_list":data})
        except:
            pass

    json.dump(output, open(output_directory, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main("./data/test1.json", "./tmp/1", "./tmp/2","./output/result.json", 1)  # Adjust the directory and number of threads as needed