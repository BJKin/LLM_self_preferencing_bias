import re
import unicodedata
from html import unescape
from pathlib import Path

import pandas as pd

from pipeline.data import RESUME_CSV, _clean, load_resumes

def clean_text(text: str) -> str:
    """ Remove invisible characters and stray separators """
    text = text.replace(chr(0xAD), "-")
    text = "".join(c for c in text if unicodedata.category(c) not in ("Cf", "Co"))
    lines = (re.sub(r" +(?:- *)?,", ",", line).strip(" ,|–—－") for line in _clean(text).split("\n"))
    return _clean("\n".join(line.removesuffix(" -") for line in lines))


def _text(html: str) -> str:
    """ Plain text of an HTML fragment with a line for each block """
    html = re.sub(r"</?(?:div|p|li|br|tr|ul|table)\b[^>]*>", "\n", re.sub(r"\s+", " ", html))
    return clean_text(unescape(re.sub(r"<[^>]*>", "", html)))


def _fields(html: str, name: str) -> list[str]:
    """ Text of each field with the given name """
    return [_text(field) for field in re.findall(FIELD.format(name) + r"(.*?)</span>", html, re.S)]


def _dates(text: str) -> list[int | None]:
    """ Dates in the text in months """
    return [None if current else int(year) * 12 + int(month or MONTHS.get(name.lower(), 6)) - 1
            for current, month, name, year in DATE.findall(text)]


def _parse_job(html: str) -> dict:
    """ Extract a job's title, bullets, and dates """
    header, _, description = html.partition('class="jobline')
    lines = _text("<" + description).split("\n") if description else []
    return {
        "title": " ".join(_fields(header, "jobtitle")),
        "bullets": [line.lstrip("-•* ") for line in lines if re.search(r"\w", line)],
        "dates": _dates(_text(re.sub(FIELD.format("jobtitle") + ".*?</span>", "", header, flags=re.S))),
    }


def _add_years(jobs: list[dict]) -> tuple[int, int] | None:
    """ Replace each job's dates with the time spent at it """
    dates = [job.pop("dates") for job in jobs]
    known = [d for job_dates in dates for d in job_dates if d is not None]
    current = max(known + [CURRENT_CUTOFF])
    for job, job_dates in zip(jobs, dates):
        job["years"] = ""
        if len(job_dates) == 2 and job_dates[0] is not None:
            start, end = (current if d is None else d for d in job_dates)
            months, years = end - start + 1, int((end - start + 1) / 12 + 0.5)
            if months > 0:
                job["years"] = "less than 1 year" if months < 12 else f"{years} year{'s' * (years > 1)}"
    if known:
        last = current if any(None in job_dates for job_dates in dates) else max(known)
        return min(known) // 12, last // 12


def _parse_education(html: str) -> tuple[str | None, dict]:
    """ Return the highest completed education group """
    graduation, attended, finished, text = {}, False, "", _text(html)
    for entry in PARAGRAPH.split(html):
        degree = " ".join(d for d in _fields(entry, "degree") if len(d.split()) <= 8)
        group = next((g for g, pattern in DEGREES.items() if re.search(pattern, degree, re.I)), None)
        if re.search(UNFINISHED, _text(entry), re.I):
            attended = attended or group not in (None, "hs_or_less")
            continue
        finished += f"\n{_text(entry)}"
        years = [d // 12 for d in _dates(" ".join(_fields(entry, "jobdates"))) or _dates(_text(entry)) if d]
        if group:
            graduation.setdefault(group, []).extend([max(years)] if years else [])
    group = next((g for g in DEGREES if g in graduation), None)
    if group in (None, "hs_or_less"):
        if attended or re.search(COLLEGE, text, re.I):
            group = "some_college"
        elif re.search(DEGREES["hs_or_less"], text, re.I):
            group = "hs_or_less"
    for higher, pattern in MENTIONED.items():
        if group == higher:
            break
        if re.search(pattern, finished, re.I):
            group = None
    return group, {g: min(years) for g, years in graduation.items() if years}


def _age_band(span: tuple[int, int] | None, edu_group: str | None, graduation: dict) -> str | None:
    """ Age band estimate using older date between first job and graduation """
    if span is None or edu_group is None:
        return None
    first, last = span
    ages = [last - first + START_AGE[edu_group]]
    ages += [last - graduation[g] + age for g, age in GRADUATION_AGE.items() if g in graduation]
    return next((band for band, (low, high) in AGE_BANDS.items() if low <= max(ages) <= high), None)


def parse_resume(html: str) -> dict:
    """ Parse resume HTML into relevant section columns """
    parts = SECTION.split(REDACTED.sub("", html))[1:]
    sections = [(code, html) for code, html in zip(parts[::2], parts[1::2]) if code not in EXCLUDED_CODES]
    untitled = [(code, TITLE.sub("", html, count=1)) for code, html in sections]
    jobs = [
        _parse_job(entry)
        for code, html in untitled if code in JOB_CODES
        for entry in PARAGRAPH.split(html) if re.search(r"\w", _text(entry))
    ]
    span = _add_years(jobs)
    edu_group, graduation = _parse_education("".join(html for code, html in untitled if code == "EDUC"))
    columns = {
        column: "\n\n".join(filter(None, (_text(html) for code, html in untitled if code in codes)))
        for column, codes in TEXT_COLUMNS.items()
    }
    named = JOB_CODES.union(*TEXT_COLUMNS.values())
    columns["other"] = "\n\n".join(filter(None, (_text(html) for code, html in sections if code not in named)))
    return {"jobs": jobs, **columns, "edu_group": edu_group, "age_band": _age_band(span, edu_group, graduation)}


def load_structured_resumes(path: Path = RESUME_CSV) -> pd.DataFrame:
    """ Load resumes into DataFrame """
    resumes = load_resumes(path).drop(columns="resume_text")
    html = pd.read_csv(Path(path), usecols=["ID", "Resume_html"], dtype=str).set_index("ID")["Resume_html"]
    parsed = pd.DataFrame([parse_resume(html[rid]) for rid in resumes["resume_id"]])
    parsed["edu_group"] = pd.Categorical(parsed["edu_group"], categories=EDU_GROUPS, ordered=True)
    parsed["age_band"] = pd.Categorical(parsed["age_band"], categories=list(AGE_BANDS), ordered=True)
    resumes = pd.concat([resumes, parsed], axis=1)
    complete = resumes["jobs"].map(lambda jobs: all(job["title"] or job["bullets"] for job in jobs))
    unwanted = resumes.apply(resume_text, axis=1).str.contains(UNWANTED)
    return resumes[complete & ~unwanted].dropna(subset=["edu_group", "age_band"]).reset_index(drop=True)


def resume_text(row, summary: str | None = None, jobs: list[dict] | None = None) -> str:
    """ Plain text of the resume shown to the screener """
    experience = []
    for original, new in zip(row.jobs, row.jobs if jobs is None else jobs, strict=True):
        header = f"{original['title']} ({original['years']})" if original["years"] else original["title"]
        experience.append("\n".join([header] + [f"- {bullet}" for bullet in new["bullets"]]).strip())
    sections = {
        "Summary": clean_text(row.human_summary if summary is None else summary),
        "Experience": "\n\n".join(filter(None, experience)),
        **{column.capitalize(): getattr(row, column) for column in [*TEXT_COLUMNS, "other"]},
    }
    return "\n\n".join(f"{heading}\n{text}" for heading, text in sections.items() if text)


# Resume section codes
JOB_CODES = {"EXPR", "WRKH", "MILI", "EEXP"}
TEXT_COLUMNS = {
    "education": {"EDUC"},
    "skills": {"SKLL", "HILT", "TSKL"},
    "accomplishments": {"ACCM", "AWAR"},
    "certifications": {"CERT"},
    "affiliations": {"AFIL"},
    "languages": {"LANG", "LNGG"},
}
EXCLUDED_CODES = {"SUMM", "NAME", "PRIN", "PICT", "PRFL", "ALNK"}

# Degree patterns
DEGREES = {
    "postgraduate": r"\bmaster|\bmagister|\bdoctor|\bjuris|\bpost[ -]?graduate|\bph\.?\s?d|\bj\.?d\b|^m\.?d\b|\bed\.?d\b"
                    r"|\be?m\.?\s?b\.?\s?a|(?<!, )\bm\.?\s?(?:s|a|s\.?c|ed|tech|f\.?a|s\.?w|p\.?a|p\.?h|h\.?a|eng)\b(?! ?word| ?office)",
    "bachelors": r"\bbat?ch[a-z]*l[a-z]*r|\bbaccalaureate|\blicenciatura"
                 r"|\bb\s?\.?\s?(?:s|a|e|s\.?c|tech|com|ed|eng|arch|b\.?a|f\.?a|s\.?n|s\.?w|a\.?s)\b|\bbs[a-z]{2}\b",
    "some_college": r"^associate|\bassociate'?s? (?:of|in|degree)|\ba\.?\s?(?:a|s|a\.?s|o\.?s)\b",
    "hs_or_less": r"\bhigh school|\bh\.?\s?s\b|\bg\.?e\.?d\b|\bsecondary|diploma|certif|licen[sc]e",
}
EDU_GROUPS = list(DEGREES)[::-1]
MENTIONED = {
    "postgraduate": r"\bmaster'?s?\b|\bm\.?b\.?a\b|\bph\.?d\b|\bdoctorate",
    "bachelors": r"\bbachelor|\bb\.\s?[sa]\b|\bbs[a-z]{0,2}\b|\bb\.?b\.?a\b",
}
COLLEGE = r"\bcollege|\buniversi|\bpolytechnic|\binstitute of technology" \
          r"|\b(?:penn|kennesaw) state|\bvirginia tech|\btexas a ?& ?m"
UNFINISHED = r"\bno degree|\bsome college|\bexpected|\banticipate|\bin[ -]pro(?:gress|cess)\b|\bpursuing|\bincomplete|\bnot complete" \
             r"|\bdid not (?:finish|complete|graduate)|\bcredit hours (?:earned|towards?)|\b(?:need|education) to (?:finish|complete)|\bcandidate for" \
             r"|\b(?:degree|phd|mba) candidate|\bcurrent student|\bwork(?:ing)? towards?|\bworking on (?:my|a)\b|\baway (?:from|for) completion" \
             r"|\btowards? (?:an?|my|the|bachelor|master|associate|degree)\b" \
             r"|\b(?:currently|presently|still) (?:enrolled|attending|pursuing|studying|a student|in (?:school|the|my))"

# Age patterns
START_AGE = {"hs_or_less": 18, "some_college": 20, "bachelors": 22, "postgraduate": 22}
GRADUATION_AGE = {"hs_or_less": 18, "bachelors": 22}
AGE_BANDS = {"18-29": (18, 29), "30-49": (30, 49), "50-64": (50, 64), "65+": (65, 80)}

# Date patterns
CURRENT_CUTOFF = 2016 * 12
MONTHS = dict(zip("jan feb mar apr may jun jul aug sep oct nov dec".split(), range(1, 13)))
DATE = re.compile(r"\b(current|present)\b|(?:(\d{1,2})/|\b([a-z]{3})[a-z]*\.?,?\s*)?\b(19[6-9]\d|20[01]\d|202[01])\b", re.I)

# Drop patterns
UNWANTED = re.compile(
    r"@\w+\.|(?i:https?:|\bwww\.)|\(?\b\d{3}\)?[-. ]\s?\d{3}[-. ]\d{4}\b|[\[\]]|_{4,}|\bCompany ?Name\b|\bCity ?, ?State\b"
)

# HTML patterns
FIELD = r'<span[^>]*class="[^"]*\b(?:{})\b[^"]*"[^>]*>'
REDACTED = re.compile(
    r"(?:<span[^>]*>[\s,|/–—－-]*</span>\s*)?" + FIELD.format("companyname|joblocation|jobcity|jobstate")
    + r"\s*(?:Company ?Name|City|State)\s*</span>"
)
SECTION = re.compile(r'<div[^>]*\bid="SECTION_([A-Z]+)[^>]*>')
TITLE = re.compile(r'<div[^>]*class="[^"]*sectiontitle.*?</div>', re.S)
PARAGRAPH = re.compile(r'(?=<div[^>]*class="[^"]*\bparagraph\b)')