dict_pre = {'key1': 'new_value1', 'key4': 'value4', 'key5': 'value5'}
dict_new = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

for key, value in dict_pre.items():
    if key in dict_new: #後の辞書のキーが前の辞書に存在する場合、後の辞書の値を優先
        dict_new[key] = value
    else: #後の辞書のキーが前の辞書に存在しない場合、新しいキーとして追加しない
        pass

print(dict_new)