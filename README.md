# LLM self-preferencing bias in hiring and its potential disparate impacts

Large Language Models (LLMs) are quickly becoming the defining technology of the decade, and as
the capabilities of these models have expanded, their judgment is increasingly being relied upon to make
decisions impacting everyday life. Applicant Tracking Systems (ATS) used to parse, filter, and rank job
applications have widely adopted these generative models to enhance the capabilities of their software and
further streamline the hiring process. As a result, processing job applications has never been more automated,
and in the initial stages of the hiring process, fewer human decisions are being made than ever before. On the
opposite side of the aisle, savvy job applicants are using LLMs to polish and fine tune their resumes based
on a given job listing, with the hopes of gaining a competitive advantage. [Xu et al.](https://arxiv.org/abs/2509.00462) showed that LLM screeners favor AI-written resumes over equally qualified human-written ones, so applicants who write their own resumes pay a penalty. But AI adoption is unequal: [Pew (2025)](https://www.pewresearch.org/short-reads/2025/06/25/34-of-us-adults-have-used-chatgpt-about-double-the-share-in-2023/) finds that 51-52% of college graduates have used ChatGPT, compared with 18% of adults with a high school education or less. Even if the screener treats every applicant identically, less-educated applicants pay this penalty more often. This begs the question: does LLM self-preferencing in hiring create disparate impact against less educated applicants?

## Repository structure
```
LLM_self_preferencing_bias/                            
├── data/       # Raw data
├── docs/       # Project documentation
├── notebooks/  # End-to-end pipeline
├── pipeline/   # Resume processing and LLM querying module
├── results/    # logs of LLM responses
└── README.MD
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/BJKin/LLM_self_preferencing_bias.git
   cd LLM_self_preferencing_bias
   ```

2. **Create and activate a virtual environment** (Python 3.10+)
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. **Install the requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API keys**
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and fill in `OPENROUTER_API_KEY` and `GEMINI_API_KEY`.