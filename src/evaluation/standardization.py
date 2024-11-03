from bs4 import BeautifulSoup
import re
import pandoc

# Aligning the Result Files Across All Models
'''
1. **Equation Alignment (Note: All LaTeX environments may include asterisks \(*\); equations must appear within a single paragraph):**
   1. Convert `\begin{equation} \end{equation}` to `\[ \]`
   2. Convert `\begin{gather} \end{gather}` to `\[ \begin{gathered} \end{gathered} \]`
   3. Apply the same transformation to `align`
   4. Convert `$$ * $$` to `\[ * \]`
   5. Convert `$ * $` to `\( * \)`
   6. Convert `\begin{multline} \end{multline}` to `\[ \]`

2. **Heading Standardization:**
   1. If a line in a paragraph is entirely composed of `---` or `===`, treat the text above it as a heading.

3. **Links and Images:**
   1. For links in the form `[.](.)`, only retain the content within `[]`.
   2. For images in the form `![]()`, remove the entire line.
   3. For images within `\begin{figure} \end{figure}`, remove the entire figure, considering the possibility of asterisks \(*\).

4. **List Standardization (Optional):**
   1. Replace any list item indicators at the beginning of a line (`+`, `-`, `*`) with a consistent symbol, adding a space after the symbol.

5. **Table Standardization:**
   1. Use `\begin{table}` to identify LaTeX-formatted tables, considering possible asterisks \(*\).
   2. Use `||` to identify GitHub-formatted tables.
   3. Only consider syntactically correct structures as tables. Convert the tables to HTML format, then standardize them into LaTeX format.
'''

def clean_md(content):
    # headings
    level_1_pattern = r'((?:[^\n]+\n)+)=+\n'
    level_2_pattern = r'((?:[^\n]+\n)+)-+\n' 
    def replace_1_match(match):
        title = match.group(1).replace('\n', ' ').strip()
        return f"# {title}\n"
    def replace_2_match(match):
        title = match.group(1).replace('\n', ' ').strip()
        return f"## {title}\n"
    content = re.sub(level_1_pattern, replace_1_match, content, flags=re.MULTILINE)
    content = re.sub(level_2_pattern, replace_2_match, content, flags=re.MULTILINE)

    # links and images
    latex_figure_pattern = re.compile(r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', re.DOTALL)
    content = latex_figure_pattern.sub('', content)
    markdown_figure_pattern = r'!\[.*?\]\(.*?\)'
    content = re.sub(markdown_figure_pattern, '', content)
    link_pattern = r'\[(.*?)\]\(.*?\)'
    content = re.sub(link_pattern, r'\1', content)

    # formulas alignment
    content = re.sub(r'\\begin\{(equation|multline)\*?\}', r'\[', content)
    content = re.sub(r'\\end\{(equation|multline)\*?\}', r'\]', content)
    content = re.sub(r'\\begin\{gather\*?\}', r'\[\n\\begin{gathered}', content)
    content = re.sub(r'\\begin\{align\*?\}', r'\[\n\\begin{aligned}', content)
    content = re.sub(r'\\end\{gather\*?\}', r'\\end{gathered}\n\]', content)
    content = re.sub(r'\\end\{align\*?\}', r'\\end{aligned}\n\]', content)
    outline_formula_pattern = r'(?<!\\)\$\$((?:(?!\n\n).)+?)(?<!\\)\$\$'
    inline_formula_pattern = r'(?<!\\)(?<!\$)\$(?!\$)((?:(?!\n\n).)+?)(?<!\\)\$(?!\$)'
    def replace_outline(match):
        return r'\[' + match.group(1) + r'\]'
    def replace_inline(match):
        return r'\(' + match.group(1) + r'\)'
    content = re.sub(outline_formula_pattern, replace_outline, content, flags=re.DOTALL)
    content = re.sub(inline_formula_pattern, replace_inline, content)

    # tables
    tables = []
    current_table_block = []
    for i, line in enumerate(content.split('\n')):
        stripped_line = line.strip()  
        if stripped_line.startswith('|') and stripped_line.endswith('|'):  
            current_table_block.append(line)
        else:  
            if current_table_block and len(current_table_block) >= 2:  
                tables.append('\n'.join(current_table_block))
            current_table_block = []  # reset current block   

    if current_table_block and len(current_table_block) >= 2:  
        tables.append('\n'.join(current_table_block))

    for table in tables:
        try:
            html_table = pandoc.write(pandoc.read(table, format='markdown'), format="html")
            latex_table = html_to_latex_table(html_table)
            content = content.replace(table, latex_table)
        except:
            continue
    
    content = re.sub(r'(\n\s*){3,}', '\n\n', content)
    return content



def html_to_latex_table(html):
    # analyze HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # extract table head
    thead = soup.find('thead')
    headers = thead.find_all('th')
    
    # set alignments dict
    alignments = {
        'right': 'r',
        'left': 'l',
        'center': 'c'
    }
    
    def get_alignment(style):
        for key in alignments:
            if key in style:
                return alignments[key]
        return 'l'  # default as left alignment
    
    # make sure alignment
    col_alignments = [get_alignment(th.get('style', '')) for th in headers]
    
    # extract table body
    tbody = soup.find('tbody')
    rows = tbody.find_all('tr')
    
    # generate LaTeX tables
    latex = "\\begin{table}\n"
    latex += "\\begin{tabular}{" + " ".join(col_alignments) + "}\n"
    latex += "\\hline\n"
    latex += " & ".join(th.get_text(strip=True) for th in headers) + " \\\\ \n"
    latex += "\\hline\n"
    
    for row in rows:
        cols = row.find_all('td')
        latex += " & ".join(col.get_text(strip=True) for col in cols) + " \\\\ \n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"
    
    return latex