# latexmk configuration for the thesis
# Run with: latexmk -pdf main.tex
$pdf_mode = 1;                       # use pdflatex
$bibtex_use = 2;                     # always run biber/bibtex
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 -file-line-error %O %S';
@default_files = ('main.tex');

# Generated/auxiliary files to clean with `latexmk -c`
$clean_ext = 'bbl bcf run.xml synctex.gz nav snm vrb';
