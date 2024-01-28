import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    

def save_json(file_path, obj):
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def load_ene2name(file_path):
    file_path = file_path
    data = load_jsonl(file_path)

    _ene2name = {}
    for d in data:
        _ene2name[d["ENE_id"]] = d["name"]["ja"]

    ene2name = {}
    for d in data:
        ids = d["ENE_id"].split(".")
        ene2name[d["ENE_id"]] = []
        for i in range(2, len(ids) + 1):
            ene2name[d["ENE_id"]].append(_ene2name[".".join(ids[:i])])

        ene2name[d["ENE_id"]] = ">".join(ene2name[d["ENE_id"]])
    return ene2name