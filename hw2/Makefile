MAIN=DIP_HW2_lupoy_D09944015_Report
APPENDIX=appendix
LATEX=xelatex
BIBTEX=bibtex
RM=rm -f

.SUFFIXES: .tex

all: $(MAIN).pdf #$(APPENDIX).pdf

$(MAIN).pdf: *.tex
	$(LATEX) $(MAIN)
	# $(BIBTEX) ${MAIN}
	$(LATEX) $(MAIN)
	# $(LATEX) $(MAIN)

#$(APPENDIX).pdf: *.tex #*.cls
#	$(LATEX) $(APPENDIX)
#	$(BIBTEX) $(APPENDIX)
#	$(LATEX) $(APPENDIX)
#	$(LATEX) $(APPENDIX)
clean:
	$(RM)  *.out *.log *.aux *.dvi *.lof *.lot *.toc *.bbl *.blg *.cut *.snm *.nav

clean-all:
	$(RM)  *.out *.log *.aux *.dvi *.lof *.lot *.toc *.bbl *.blg *.cut *.snm *.nav ${MAIN}.pdf 
