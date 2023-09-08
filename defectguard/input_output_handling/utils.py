def extract_owner_and_repo(commit_link):
    # Split the link by '/' character
    parts = commit_link.split('/')

    # Get the owner and repository name from the parts
    owner = parts[3]
    repo_name = parts[4]
    commit_hash = parts[6]

    # Return the owner and repository name
    return owner, repo_name, commit_hash

def extract_diff(diff):
    num_added_lines = 0
    list_file_changes = []
    for file_elem in list(diff.items()):
        file_path = file_elem[0]
        file_val = file_elem[1]
            
        file = {"file_name": file_path, "code_changes":[]}
        for ab in file_val["content"]:
            if "ab" in ab:
                continue
            hunk = {"added_code":[], "removed_code":[]}
            if "a" in ab:
                hunk["removed_code"] += [line.strip() for line in ab["a"]]
            if "b" in ab:
                hunk["added_code"] += [line.strip() for line in ab["b"]]
                num_added_lines += len(ab["b"])
            hunk["added_code"] = "\n".join(hunk["added_code"])
            hunk["removed_code"] = "\n".join(hunk["removed_code"])
            file["code_changes"].append(hunk)
        list_file_changes.append(file)
    return list_file_changes, num_added_lines

def split_sentence(sentence):
    sentence = sentence.replace('.', ' . ').replace('_', ' ').replace('@', ' @ ')\
        .replace('-', ' - ').replace('~', ' ~ ').replace('%', ' % ').replace('^', ' ^ ')\
        .replace('&', ' & ').replace('*', ' * ').replace('(', ' ( ').replace(')', ' ) ')\
        .replace('+', ' + ').replace('=', ' = ').replace('{', ' { ').replace('}', ' } ')\
        .replace('|', ' | ').replace('\\', ' \ ').replace('[', ' [ ').replace(']', ' ] ')\
        .replace(':', ' : ').replace(';', ' ; ').replace(',', ' , ').replace('<', ' < ')\
        .replace('>', ' > ').replace('?', ' ? ').replace('/', ' / ')
    sentence = ' '.join(sentence.split())
    return sentence

def commit_to_info(commit):
    list_file_changes, num_added_lines = extract_diff(commit["diff"])
    
    return {
            'commit_hash': commit["commit_id"],
            'commit_message': commit['commit_msg'],
            'main_language_file_changes': list_file_changes,
            'num_added_lines_in_main_language': num_added_lines,
        }

def hunks_to_code(file_levels: list) -> str:
    code = []
    for file_level in file_levels:
        for hunk in file_level['code_changes']:
            added_code = hunk['added_code']
            removed_code = hunk['removed_code']

            added_code = added_code.strip()
            removed_code = removed_code.strip()

            added_code = ' '.join(split_sentence(added_code).split())
            removed_code = ' '.join(split_sentence(removed_code).split())

            added_code = ' '.join(added_code.split(' '))
            removed_code = ' '.join(removed_code.split(' '))

            code.append(added_code)
            code.append(removed_code)

    return code

def diff_to_code_change(diff):
    list_hunks = []
    for file_val in diff.values():
        for ab in file_val["content"]:
            if "ab" in ab:
                continue
            hunk = {"added_code":[], "removed_code":[]}
            if "a" in ab:
                hunk["removed_code"] += [line.strip() for line in ab["a"]]
            if "b" in ab:
                hunk["added_code"] += [line.strip() for line in ab["b"]]
            list_hunks.append(hunk)
    return list_hunks

def commit_to_code_change(commit):
    
    return {
        "commit id": [commit["commit_id"]],
        "code change": [diff_to_code_change(commit["diff"])]
    }
    