import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from pipeline import client, log

CALL_KEYS = ("model", "prompt", "temperature")


def run_jobs(jobs: list[dict], log_path: Path, workers: int = 4) -> None:
    """ Run each job and log its response (jobs already in the log are skipped) """
    done = set(log.load_log(log_path)["job_id"]) if Path(log_path).exists() else set()
    todo = [job for job in jobs if job["job_id"] not in done]
    lock = threading.Lock()

    def run(job: dict) -> None:
        resp = client.generate(job["model"], job["prompt"], job["temperature"])
        with lock:
            log.log_response(log_path, resp, **{k: v for k, v in job.items() if k not in CALL_KEYS})

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(run, job) for job in todo]
        for future in tqdm(as_completed(futures), total=len(todo), desc=f"{len(jobs) - len(todo)} already done"):
            future.result()
    finally:
        pool.shutdown(cancel_futures=True)
