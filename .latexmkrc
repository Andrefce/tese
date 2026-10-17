# latexmk configuration for the thesis
# Run with: latexmk main.tex
$pdf_mode = 5;                       # use xelatex
$bibtex_use = 2;                     # always run biber/bibtex
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 -file-line-error %O %S';
# Default xdvipdfmx compression (-z 9) spends ~35 s on the PNG figures for
# barely smaller output; -z 6 is about 4x faster.
$xdvipdfmx = 'xdvipdfmx -z 6 -E -o %D %O %S';
@default_files = ('main.tex');

# Generated/auxiliary files to clean with `latexmk -c`
$clean_ext = 'bbl bcf run.xml synctex.gz nav snm vrb';
