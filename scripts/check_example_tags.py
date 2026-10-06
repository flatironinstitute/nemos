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
    "topic": [
        "feature-design",
        "variable-selection",
        "cross-validation",
        "regularization",
        "stochastic-fit",
        "scalability",
        "functional-connectivity",
        "simulation",
        "custom-components",
        "receptive-fields",
        "latent-states",
    ],
    "data": ["recorded", "simulated"],
}

# required alongside EXPECTED_TAGS, but free text rather than a vocabulary
DESCRIPTION_FIELD = "description"
TAG_FIELDS = EXPECTED_TAGS.keys() | {DESCRIPTION_FIELD}


EXAMPLE_DIRS = ("tutorials", "how_to_guide")


def get_tags(md_text):
    metadata = read_metadata(md_text, "md")
    return metadata.get("nemos_tags", None)


def get_invalid_and_missing_tag_fields(metadata):
    invalid = metadata.keys() - TAG_FIELDS
    missing = TAG_FIELDS - metadata.keys()
    return list(invalid), list(missing)


def has_empty_description(metadata):
    if DESCRIPTION_FIELD not in metadata:
        return False
    description = metadata[DESCRIPTION_FIELD]
    return not isinstance(description, str) or not description.strip()


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
    # Find files where jupytext successfully detects a MyST format
    no_tags = []
    invalid_tag_fields = []
    invalid_tag_entries = []
    missing_tag_fields = []
    empty_description = []
    for p in docs_dir.glob("**/*.md"):
        if ".ipynb_checkpoints" in p.parts:
            continue
        with open(p) as f:
            text = f.read()
            fmt = jupytext.guess_format(text, ".md")[0]
            if fmt == "myst":
                nemos_tags = get_tags(text)
                if nemos_tags is None:
                    if set(p.relative_to(docs_dir).parts) & set(EXAMPLE_DIRS):
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
                    if has_empty_description(nemos_tags):
                        empty_description.append(p)
    return (
        no_tags,
        invalid_tag_fields,
        missing_tag_fields,
        invalid_tag_entries,
        empty_description,
    )


if __name__ == "__main__":
    import logging
    import sys

    logger = logging.getLogger("check_example_tags")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    no_tag, inv_tag_fields, missing_tags_fields, inv_tag_entries, empty_desc = main()
    msg = ""
    for path in no_tag:
        msg += f"{path}\nNo `nemos_tags` found in metadata\n"

    for path, fields in inv_tag_fields:
        msg += f"\n{path}:\n\tInvalid `nemos_tag` field names: {fields}\n\tAvailable fields: {sorted(TAG_FIELDS)}\n"

    for path, fields in missing_tags_fields:
        msg += f"\n{path}:\n\tMissing `nemos_tag` fields: {fields}\n"

    for path, fields in inv_tag_entries:
        for field_name, entries in fields.items():
            msg += f"\n{path}:\n\tInvalid `nemos_tag` entries for field '{field_name}': {entries}\n\tAvailable entries for '{field_name}': {list(EXPECTED_TAGS[field_name])}\n"

    for path in empty_desc:
        msg += f"\n{path}:\n\t`nemos_tag` field '{DESCRIPTION_FIELD}' must be a non-empty string\n"

    if msg:
        logger.warning(msg)
        sys.exit(1)
    logger.info("No nemos_tag inconsistencies found.")
