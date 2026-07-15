"""Single-worker background jobs for long-running ASR, LLM, and render tasks."""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    kind: str
    status: str = "queued"
    message: str = "等待后台工作线程。"
    result: Any = None
    error: str | None = None
    progress: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="funclip-job")

    def submit(self, kind: str, worker: Callable[[Callable[[str, int | None], None]], Any]) -> str:
        job_id = uuid.uuid4().hex
        job = Job(kind=kind)
        with self._jobs_lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, job, worker)
        return job_id

    def _run(self, job_id: str, job: Job, worker: Callable[[Callable[[str, int | None], None]], Any]) -> None:
        def report(message: str, progress: int | None = None) -> None:
            with job.lock:
                job.status = "running"
                job.message = message
                if progress is not None:
                    job.progress = max(job.progress, min(100, int(progress)))
            logging.warning("后台任务 %s [%s]：%s", job_id[:8], job.kind, message)

        stop_pulse = threading.Event()

        def pulse() -> None:
            while not stop_pulse.wait(4):
                with job.lock:
                    if job.status == "running" and job.progress < 90:
                        job.progress += 1

        try:
            report("后台任务已开始。")
            threading.Thread(target=pulse, name="funclip-progress", daemon=True).start()
            result = worker(report)
            with job.lock:
                job.status = "completed"
                job.message = "任务完成。"
                job.progress = 100
                job.result = result
        except Exception as exc:
            logging.error("后台任务 %s [%s] 失败：\n%s", job_id[:8], job.kind, traceback.format_exc())
            with job.lock:
                job.status = "failed"
                job.message = "任务失败。"
                job.error = str(exc)
        finally:
            stop_pulse.set()

    def get(self, job_id: str | None) -> dict[str, Any] | None:
        if not job_id:
            return None
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        with job.lock:
            return {
                "kind": job.kind,
                "status": job.status,
                "message": job.message,
                "result": job.result,
                "error": job.error,
                "progress": job.progress,
            }


JOBS = JobManager()
