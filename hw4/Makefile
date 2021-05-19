MAIN=DIP_HW4_lupoy_D09944015_Report
LATEX=xelatex
BIBTEX=bibtex
RM=rm -f

.SUFFIXES: .tex

all: $(MAIN).pdf

$(MAIN).pdf: *.tex
	$(LATEX) $(MAIN)
	# $(BIBTEX) ${MAIN}
	$(LATEX) $(MAIN)
	# $(LATEX) $(MAIN)

clean:
	$(RM)  *.out *.log *.aux *.dvi *.lof *.lot *.toc *.bbl *.blg *.cut *.snm *.nav

clean-all:
	$(RM)  *.out *.log *.aux *.dvi *.lof *.lot *.toc *.bbl *.blg *.cut *.snm *.nav ${MAIN}.pdf 
