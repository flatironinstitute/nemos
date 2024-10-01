"""Generate the code reference pages and navigation.

See [CCN template repo](https://ccn-template.readthedocs.io/en/latest/notes/03-documentation/) for why.
"""

from pathlib import Path
import re
import mkdocs_gen_files

SKIP_MODULES = ("styles", "_documentation_utils", "_regularizer_builder")


def skip_module(module_path: Path):
    return any(p in SKIP_MODULES for p in module_path.with_suffix("").parts)


def filter_nav(iter_literate_nav):
    filtered_nav = []
    for line in iter_literate_nav:
        if not any(re.search(rf"\[{p}]", line) for p in SKIP_MODULES):
            filtered_nav.append(line)
    return filtered_nav


nav = mkdocs_gen_files.Nav()

for path in sorted(Path("src").rglob("*.py")):

    module_path = path.relative_to("src").with_suffix("")

    if skip_module(module_path):
        continue

    doc_path = path.relative_to("src").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    # if the md file name is `module.md`, generate documentation from docstrings
    if full_doc_path.name != 'index.md':
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}")

    # if the md file name is `index.md`, add the list of modules with hyperlinks
    else:
        this_module_path = Path("src") / Path(*parts)
        module_index = ""
        for module_scripts in sorted(this_module_path.rglob("*.py")):

            if "__init__" in module_scripts.name:
                continue
            elif skip_module(module_scripts):
                continue

            tabs = ""
            cumlative_path = []
            for i, p in enumerate(module_scripts.parts[len(this_module_path.parts):]):
                cumlative_path.append(p)
                relative_path = Path(*cumlative_path)
                module_index += tabs + (f"* [{p.replace('.py', '')}]"
                                        f"({relative_path.as_posix().replace('.py', '.md')})\n")
                tabs = "\t" + tabs

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write(module_index)




    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:

    literate_nav = nav.build_literate_nav()

    # Filter out private modules and the styles directory
    nav_file.writelines(filter_nav(literate_nav))