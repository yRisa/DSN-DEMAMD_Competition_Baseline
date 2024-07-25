import json

#请修改prompt
prompt = '请抽取因果关系事件对'

def list2jsonl(list_):
    return_str=""
    system="你是一个因果分析专家，擅长发现文本中的因果关系。"
    for i,item in enumerate(list_):
        print(i)
        q=item[0]
        a=item[1]
        tmp_dict={"type": "chatml","messages": [{"role": "system","content": system},{"role": "user","content": q},{"role": "assistant","content": a}],"source": "unknown"}
        return_str += json.dumps(tmp_dict,ensure_ascii=False) + "\n"
    return return_str

def func(input_path,input_path2, output_path):
    tmp_list=[]

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(input_path2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    #print(data)
    for item in data:
        print(item)
        q=item["text"]+prompt
        a=json.dumps(item["causality_list"],ensure_ascii=False,indent=4)
        tmp_list.append([q,a])

    for item in data2:
        print(item)
        q=item["text"]+prompt
        a=json.dumps(item["causality_list"],ensure_ascii=False,indent=4)
        tmp_list.append([q,a])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(list2jsonl(tmp_list))


if __name__ == "__main__":
    func("data/train1.json","data/train2.json", "sft/data/1.jsonl")