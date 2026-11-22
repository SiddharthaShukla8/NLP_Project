#!/usr/bin/env python3
"""
Convert Markdown milestone documentation to high-quality, professional PDF reports.
Uses markdown2 with extra extensions and xhtml2pdf with custom academic styling.
"""

import os
import re
import markdown2
from xhtml2pdf import pisa

CSS = """
@page {
    size: a4 portrait;
    margin: 20mm 15mm 20mm 15mm;
    @bottom-right {
        content: "Page " counter(page);
        font-size: 8.5pt;
        color: #757575;
    }
    @bottom-left {
        content: "NLP Research Project | RAG Robustness";
        font-size: 8.5pt;
        color: #757575;
    }
}
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.5;
    color: #212121;
}
h1 {
    color: #1a237e;
    font-size: 16pt;
    border-bottom: 2px solid #1a237e;
    padding-bottom: 4px;
    margin-top: 0;
    margin-bottom: 10px;
}
h2 {
    color: #283593;
    font-size: 13pt;
    margin-top: 14px;
    margin-bottom: 6px;
    border-bottom: 1px solid #c5cae9;
    padding-bottom: 2px;
}
h3 {
    color: #3949ab;
    font-size: 10.5pt;
    margin-top: 10px;
    margin-bottom: 4px;
}
p {
    margin-top: 3px;
    margin-bottom: 6px;
}
table {
    width: 100%;
    margin: 8px 0;
    font-size: 8pt;
}
th {
    background-color: #1a237e;
    color: #ffffff;
    font-weight: bold;
    padding: 5px 6px;
    text-align: left;
    border: 1px solid #1a237e;
}
td {
    padding: 4px 6px;
    border: 1px solid #dcdcdc;
    vertical-align: top;
}
tr:nth-child(even) td {
    background-color: #f8f9fa;
}
code {
    font-family: Courier, monospace;
    background-color: #f1f3f4;
    padding: 1px 3px;
    font-size: 8pt;
    color: #ad1457;
}
pre {
    background-color: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 3px;
    padding: 6px 8px;
    font-family: Courier, monospace;
    font-size: 7.5pt;
    line-height: 1.25;
}
blockquote {
    border-left: 3px solid #1a237e;
    padding: 6px 10px;
    background-color: #e8eaf6;
    margin: 6px 0;
    font-size: 9pt;
}
hr {
    border: 0;
    height: 1px;
    background: #e0e0e0;
    margin: 10px 0;
}
ul, ol {
    margin-top: 3px;
    margin-bottom: 6px;
    padding-left: 18px;
}
li {
    margin-bottom: 2px;
}
"""

def clean_markdown_for_pdf(text):
    text = text.replace("█", "#")
    # Format math formulas into readable typography
    text = re.sub(r'\$\$([\s\S]*?)\$\$', r'<p style="text-align:center;font-weight:bold;margin:6px 0;">\1</p>', text)
    text = re.sub(r'\$([^\$\n]+?)\$', r'<b>\1</b>', text)
    return text

def convert_file(md_path, pdf_path):
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return False
    
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    cleaned_text = clean_markdown_for_pdf(md_text)
    html_content = markdown2.markdown(cleaned_text, extras=['tables', 'fenced-code-blocks', 'header-ids'])
    full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_content}</body></html>"

    with open(pdf_path, 'wb') as f_out:
        pisa_status = pisa.CreatePDF(full_html, dest=f_out)

    if pisa_status.err == 0:
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"Generated: {pdf_path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"Error generating {pdf_path}")
        return False

if __name__ == "__main__":
    docs = [
        ("docs/M1_Literature_Survey/M1_Literature_Survey_Report.md", "docs/M1_Literature_Survey/M1_Literature_Survey_Report.pdf"),
        ("docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.md", "docs/M2_Baseline_Reproduction/M2_Baseline_Reproduction_Report.pdf"),
        ("docs/M3_Research_Gap_Hypothesis/M3_Research_Gap_Hypothesis_Report.md", "docs/M3_Research_Gap_Hypothesis/M3_Research_Gap_Hypothesis_Report.pdf"),
        ("docs/M4_Controlled_Experiments/M4_Controlled_Experiments_Report.md", "docs/M4_Controlled_Experiments/M4_Controlled_Experiments_Report.pdf"),
        ("docs/M5_Failure_Analysis/M5_Failure_Analysis_Report.md", "docs/M5_Failure_Analysis/M5_Failure_Analysis_Report.pdf"),
        ("docs/M6_Final_Paper/M6_Final_IEEE_Paper.md", "docs/M6_Final_Paper/M6_Final_IEEE_Paper.pdf")
    ]
    for md, pdf in docs:
        convert_file(md, pdf)
