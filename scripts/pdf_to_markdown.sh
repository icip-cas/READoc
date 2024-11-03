#!/bin/bash

if [ "$SYSTEM" == *"pymupdf4llm"* ]; then
    python src/pdf_to_markdown/baseline_pymupdf4llm.py --pdf_dir data/$SUBSET/pdf --md_dir output/$SUBSET/$SYSTEM
elif [ "$SYSTEM" == *"marker"* ]; then
    python src/pdf_to_markdown/pipeline_marker.py --pdf_dir data/$SUBSET/pdf --md_dir output/$SUBSET/$SYSTEM
elif [ "$SYSTEM" == *"nougat"* ]; then
    python src/pdf_to_markdown/expert_nougat.py -o output/$SUBSET/$SYSTEM data/$SUBSET/pdf
elif [ "$SYSTEM" == *"internvl-chat-v1-5"* ]; then
    if [ "$SUBSET" == *"arxiv"* ]; then
        prompt="This image displays a document page, convert the page content into Markdown format. Use continuous # to denote headings at each level. Display tables following markdown or latex format. Use $ or \(\) to surround inline math, and use \$\$ or \[\] to surround isolated math block. Don't explain, directly output the Markdown-format content."
    else
        prompt="This image displays a document page, convert the page content into Markdown format. Use continuous # to denote headings at each level. If it's a blank page, don't output anything. Otherwise, directly output the Markdown-format content without explanation."
    fi
    python src/pdf_to_markdown/vlm_internvl-chat-v1-5.py --pdf_dir data/$SUBSET/pdf --md_dir output/$SUBSET/$SYSTEM --prompt $prompt
elif [ "$SYSTEM" == *"gpt-4o-mini"* ]; then
    if [ "$SUBSET" == *"arxiv"* ]; then
        prompt="This image displays a document page, convert the page content into Markdown format. Use continuous # to denote headings at each level. Display tables following markdown or latex format. Use $ or \(\) to surround inline math, and use \$\$ or \[\] to surround isolated math block. Don't explain, directly output the Markdown-format content."
    else
        prompt="This image displays a document page, convert the page content into Markdown format. Use continuous # to denote headings at each level. If it's a blank page, don't output anything. Otherwise, directly output the Markdown-format content without explanation."
    fi
    python src/pdf_to_markdown/vlm_gpt-4o-mini.py --pdf_dir data/$SUBSET/pdf --md_dir output/$SUBSET/$SYSTEM --prompt $prompt
else
    echo "Error. Not Implement Yet."
fi