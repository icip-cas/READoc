import re


# After Standardization, segment the documents (extract useful materials for further evaluation)
def extract_materials(md_content):
    
    inline_math_regex = re.compile(r"\\\(((.*?)(?<!\\))\\\)")
    outline_math_regex = re.compile(r"\\\[((.+?)(?<!\\))\\\]", re.S)
    table_regex = re.compile(r"(\n\\begin\{table\}((?:(?!\\begin\{table\}).)*?)\\end\{table\})", re.S)
    head_regex = re.compile(r'^((#{1,6}) +(.+))', re.MULTILINE)

    matches = []
    tables = []
    heads = []
    inline_math = []
    outline_math = []
    
    # extract tables, headings, formulas
    for match in head_regex.finditer(md_content):
        matches.append((match.start(), match.end(), 'head', match.group()))
        heads.append(match.group())
    for match in table_regex.finditer(md_content):
        matches.append((match.start(), match.end(), 'table', match.group()))
        tables.append(match.group())
    for match in outline_math_regex.finditer(md_content):
        matches.append((match.start(), match.end(), 'math', match.group()))
        outline_math.append(match.group()) 
    for match in inline_math_regex.finditer(md_content):
        inline_math.append(match.group()) 

    # remove the segmented content, remaining parts are treated as plain text
    md_content_plain = table_regex.sub("", md_content)
    md_content_plain = head_regex.sub("", md_content_plain)
    md_content_plain = inline_math_regex.sub("", md_content_plain)
    md_content_plain = outline_math_regex.sub("", md_content_plain)
    md_content_plain = re.sub(r'\n{3,}', '\n\n', md_content_plain)

    # Create a list to sequentially store all parts
    matches.sort(key=lambda x: x[0])
    segments = []
    pos = 0
    for start, end, match_type, content in matches:
        if start < pos:
            continue
        if start > pos:
            for paragraph in md_content[pos:start].strip().split('\n\n'):
                segments.append(paragraph)

        segments.append(content)
        pos = end
    
    if pos < len(md_content):
        for paragraph in md_content[pos:].strip().split('\n\n'):
            segments.append(paragraph)

    segments = [segment.strip() for segment in segments if segment.strip() != '']

    json_content = {'overall': md_content.strip(), 'segments': segments,
        'subtask':{'plain': md_content_plain.strip(), 'table': tables, 'heads': heads, 'math': {'inline': inline_math, 'outline': outline_math}}}
    
    return json_content
