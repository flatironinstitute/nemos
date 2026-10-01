from pathlib import Path

import jupytext
from jupytext.formats import read_metadata

EXPECTED_TAGS = {
    "signal": [
        "spike-counts",
        "calcium-imaging",
        "lfp",
        "behavior-choices",
        "behavior-tracking",
        "continuous",
    ],
    "observation_model": [
        "Poisson",
        "Gamma",
        "Gaussian",
        "Bernoulli",
        "NegativeBinomial",
        "Categorical",
    ],
    "model": ["GLM", "PopulationGLM", "GLMHMM", "ClassifierGLM"],
    "data": ["recorded", "simulated"],
}


def get_tags(md_text):
    metadata = read_metadata(md_text, "md")
    return metadata.get("nemos_tags", None)


def get_invalid_and_missing_tag_fields(metadata):
    invalid = metadata.keys() - EXPECTED_TAGS.keys()
    missing = EXPECTED_TAGS.keys() - metadata.keys()
    return list(invalid), list(missing)


def get_invalid_tag_entries(metadata):
    invalid = {}
    for field_name in EXPECTED_TAGS.keys():
        entry_values = metadata.get(field_name, [])
        entry_values = (
            entry_values if isinstance(entry_values, list) else [entry_values]
        )
        invalid_entries = set(entry_values) - set(EXPECTED_TAGS[field_name])
        if invalid_entries:
            invalid[field_name] = (
                [invalid_entries]
                if isinstance(invalid_entries, str)
                else list(invalid_entries)
            )
    return invalid


def main():
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    print(docs_dir)
    # Find files where jupytext successfully detects a MyST format
    no_tags = []
    invalid_tag_fields = []
    invalid_tag_entries = []
    missing_tag_fields = []
    for p in docs_dir.glob("**/*.md"):
        with open(p) as f:
            text = f.read()
            fmt = jupytext.guess_format(text, ".md")[0]
            if fmt == "myst":
                nemos_tags = get_tags(text)
                if nemos_tags is None:
                    no_tags.append(p)
                else:
                    invalid_fields, missing_fields = get_invalid_and_missing_tag_fields(
                        nemos_tags
                    )
                    invalid_entries = get_invalid_tag_entries(nemos_tags)
                    if invalid_fields:
                        invalid_tag_fields.append((p, invalid_fields))
                    if invalid_entries:
                        invalid_tag_entries.append((p, invalid_entries))
                    if missing_fields:
                        missing_tag_fields.append((p, missing_fields))
    return invalid_tag_fields, missing_tag_fields, invalid_tag_entries


if __name__ == "__main__":
    import logging
    import sys

    logger = logging.getLogger("check_example_tags")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    inv_tag_fields, missing_tags_fields, inv_tag_entries = main()
    msg = ""
    for path, fields in inv_tag_fields:
        msg += f"\n{path}:\n\tInvalid `nemos_tag` field names: {fields}\n\tAvailable fields: {list(EXPECTED_TAGS.keys())}\n"

    for path, fields in missing_tags_fields:
        msg += f"\n{path}:\n\tMissing `nemos_tag` fields: {fields}\n"

    for path, fields in inv_tag_entries:
        for field_name, entries in fields.items():
            msg += f"\n{path}:\n\tInvalid `nemos_tag` entries for field '{field_name}': {entries}\n\tAvailable entries for '{field_name}': {list(EXPECTED_TAGS[field_name])}\n"

    if msg:
        logger.warning(msg)
        sys.exit(1)
    logger.info("No nemos_tag inconsistencies found.")
