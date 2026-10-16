"""Index the tagged examples, and link to them filtered by tag.

Every tutorial and how-to guide carries a ``nemos_tags`` block in its
frontmatter (see ``developers_notes/10-example_tags.md``). This extension

- writes those blocks to ``_static/examples.json``, which the Example Finder
  page filters;
- provides the ``nemos-examples`` directive, a link to the Example Finder with
  the filters given as options, e.g. ``:model: GLM, PopulationGLM``.
"""

import json
from pathlib import Path
from urllib.parse import quote

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

FIELDS = ("model", "observation_model", "signal", "topic", "data")


def write_examples_index(app, exception):
    """Collect the ``nemos_tags`` of every example into ``_static/examples.json``.

    Written into the build rather than the sources, so it is never out of step
    with the pages it links to.
    """
    if exception is not None:
        return
    env = app.env
    examples = []
    for docname in sorted(env.found_docs):
        tags = env.metadata[docname].get("nemos_tags")
        if tags is None:
            continue
        examples.append(
            {
                "title": env.titles[docname].astext(),
                "url": app.builder.get_target_uri(docname),
                # myst_parser hands nested frontmatter over as a JSON string
                **json.loads(tags),
            }
        )
    out = Path(app.outdir) / "_static" / "examples.json"
    out.write_text(json.dumps(examples, indent=2))


class ExamplesDirective(SphinxDirective):
    """Link to the Example Finder, filtered by the options."""

    option_spec = {field: str for field in FIELDS}

    def run(self):
        # values in a field are comma-separated, as in the Example Finder's address
        query = "&".join(
            f"{field}={','.join(quote(v.strip()) for v in values.split(','))}"
            for field, values in self.options.items()
        )
        uri = self.env.app.builder.get_relative_uri(self.env.docname, "examples")
        link = nodes.reference(
            "", "See the examples in the Example Finder", refuri=f"{uri}?{query}"
        )
        return [nodes.paragraph("", "", link)]


def setup(app):
    app.add_directive("nemos-examples", ExamplesDirective)
    app.connect("build-finished", write_examples_index)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
