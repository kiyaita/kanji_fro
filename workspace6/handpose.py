import json

def select_list(fruit1, fruit2):
    # リストと対応する答えを読み込みます
    with open("workspace6\list_answers.json", "r", encoding='utf-8') as f:
        list_answers = json.load(f)
    print(list_answers)
    fruitsA = fruit1 +","+ fruit2
    fruitsB = fruit2 +","+ fruit1
    # フルーツの組み合わせに対応するリストを返します
    if fruitsA in list_answers:
        return list_answers[fruitsA]
    elif fruitsB in list_answers:
        return list_answers[fruitsB]
    else:
        return "None"



print(select_list("ぞう","りんご"))
print(select_list("りんご","ぞう"))
print(select_list("ぞう","ぞう"))