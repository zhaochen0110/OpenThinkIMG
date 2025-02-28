def extract_nested_json(text):
    stack = [] 
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if not stack: 
                start = i
            stack.append("{")  
        elif char == "}":
            stack.pop() 
            if not stack: 
                try:
                    json_str = text[start:i+1]
                    json_cleaned = remove_comments_from_json(json_str)
                    return json.loads(json_cleaned)
                except json.JSONDecodeError as e:
                    continue 
    return None 