# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
# Set an environ variable available during sphinx build
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile fullapi

# Local-only build that documents every module, class and function -- private
# members included -- under api/full.html. Notebooks are not executed, so this
# is also the quickest way to iterate on API docstrings.
fullapi:
	NEMOS_FULL_API=true EXCLUDE_TUTORIALS=true \
		$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)/fullapi" $(SPHINXOPTS) $(O)
	@echo "Open $(BUILDDIR)/fullapi/html/api/full.html"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
