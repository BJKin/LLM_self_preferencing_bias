import re
import shutil
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

import kagglehub
import pandas as pd

from pipeline.config import DATA

RESUME_COLUMNS = ["resume_id", "category", "resume_text", "human_summary"]

# Kaggle datasets
DATASETS = {
    "resumes": ("snehaanbhawal/resume-dataset", ["Resume/Resume.csv"], DATA),
    "jobs": ("arshkon/linkedin-job-postings", [
        "postings.csv", "jobs/job_skills.csv", "jobs/job_industries.csv", "mappings/skills.csv", "mappings/industries.csv",
    ], DATA / "LinkedIn"),
}
RESUME_CSV = DATA / "Resume" / "Resume.csv"
JOBS_DIR = DATA / "LinkedIn"

# Remove resumes with summaries which are clearly junk
JUNK_SUMMARY_IDS = {
    "90867631",  
    "18187364", 
    "37087371",
}


def download_data(force: bool = False) -> None:
    """ Download the resume and job posting CSVs from Kaggle into ./data/ """
    for handle, files, folder in DATASETS.values():
        for file in files:
            if force or not (folder / file).exists():
                kagglehub.dataset_download(handle, path=file, output_dir=str(folder), force_download=force)
        # remove Kagglehub's download finished markers
        shutil.rmtree(folder / ".complete", ignore_errors=True)


class _SectionParser(HTMLParser):
    """ Collect the heading and text of each resume section from a given HTML """

    BLOCK_TAGS = {"div", "p", "li", "br", "tr", "ul", "table"}

    def __init__(self):
        super().__init__()
        self.sections = []  
        self.depth = 0     
        self.in_heading = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            match = re.match(r"SECTION_([A-Z]+)", attrs.get("id") or "")
            if match and self.depth == 0:
                self.sections.append([match.group(1), "", []])
                self.depth = 1
                return
            if self.depth:
                self.depth += 1
                self.in_heading = "sectiontitle" in (attrs.get("class") or "")
        if self.depth and tag in self.BLOCK_TAGS:
            self.sections[-1][2].append("\n")

    def handle_endtag(self, tag):
        if tag == "div" and self.depth:
            self.depth -= 1
            self.in_heading = False
        if self.depth and tag in self.BLOCK_TAGS:
            self.sections[-1][2].append("\n")

    def handle_data(self, data):
        if self.depth and self.in_heading:
            self.sections[-1][1] += data
        elif self.depth:
            self.sections[-1][2].append(data)


def _clean(text: str) -> str:
    """ Remove odd characters, repeated spaces and blank lines """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n[ \n]*", "\n", text)
    return text.strip()


def _split_resume(html: str) -> tuple[str, str]:
    """ Split summary from rest of resume text """
    parser = _SectionParser()
    parser.feed(html)
    summary, rest = [], []
    for code, heading, parts in parser.sections:
        heading, text = _clean(heading), _clean("".join(parts))
        if code == "SUMM":
            summary.append(" ".join(text.split())) 
        elif text:
            rest.append(f"{heading}\n{text}" if heading else text)
    return " ".join(summary), "\n\n".join(rest)


def load_resumes(path: Path = RESUME_CSV) -> pd.DataFrame:
    """ Load resume .csv, extract summary/body text from HTML, and drop resumes where the summary is missing """
    raw = pd.read_csv(Path(path))
    split = raw["Resume_html"].map(_split_resume)
    resumes = pd.DataFrame({
        "resume_id": raw["ID"].astype(str),
        "category": raw["Category"],
        "resume_text": split.str[1],
        "human_summary": split.str[0],
    })
    keep = (
        (resumes["human_summary"] != "")
        & (resumes["resume_text"] != "")
        & ~resumes["resume_id"].isin(JUNK_SUMMARY_IDS)
    )
    return resumes[keep].reset_index(drop=True)[RESUME_COLUMNS]
