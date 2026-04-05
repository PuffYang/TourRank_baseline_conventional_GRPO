import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RubricItem:
    rubric_item: str
    type: str


@dataclass
class ResearchQAItem:
    id: str
    general_domain: str
    subdomain: str
    field: str
    query: str
    date: str
    rubric: List[RubricItem]


DATA_DIR = Path(__file__).resolve().parent / "data"


def download_researchqa_dataset(
    split: str = "test.json", output_dir: str = "evaluation/research_qa_eval/data"
) -> str:
    candidate_dirs: list[Path] = []
    override_dir = os.environ.get("RESEARCHQA_DATA_DIR")
    if override_dir:
        candidate_dirs.append(Path(override_dir).expanduser())
    if output_dir:
        candidate_dirs.append(Path(output_dir).expanduser())
    candidate_dirs.append(DATA_DIR)

    for candidate_dir in candidate_dirs:
        candidate_path = candidate_dir / split
        if candidate_path.exists():
            return str(candidate_path)

    if os.environ.get("RESEARCHQA_DOWNLOAD_MISSING_DATA", "0") != "1":
        searched = [str(candidate_dir / split) for candidate_dir in candidate_dirs]
        raise FileNotFoundError(
            "ResearchQA evaluation data not found locally. "
            f"Sync {split} under {DATA_DIR} or set RESEARCHQA_DATA_DIR. "
            f"Searched: {searched}"
        )

    target_dir = Path(output_dir).expanduser() if output_dir else DATA_DIR
    output_path = target_dir / split
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    file_path = hf_hub_download(
        repo_id="realliyifei/ResearchQA",
        filename=split,
        repo_type="dataset",
        revision="87cdd81df0c5ea96de293859233e8e64dac3d168",
    )
    shutil.copy(file_path, output_path)

    return str(output_path)


def load_researchqa_data(json_path: str) -> List[ResearchQAItem]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for item in data:
        rubric = [RubricItem(**r) for r in item["rubric"]]
        items.append(
            ResearchQAItem(
                id=item["id"],
                general_domain=item["general_domain"],
                subdomain=item["subdomain"],
                field=item["field"],
                query=item["query"],
                date=item["date"],
                rubric=rubric,
            )
        )
    return items
