import asyncio
import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class Job:
    args: tuple
    kwargs: dict
    target: callable


def create_job(target: callable, *args, **kwargs):
    return Job(target=target, args=args, kwargs=kwargs)


class Worker:

    def __init__(self, name):
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.name = name
        self._queue = queue.SimpleQueue()
        self._results = queue.SimpleQueue()

        self._stop = False

    def start(self):
        self._thread.start()

    def enqueue(self, job):
        self._queue.put_nowait(job)

    async def wait_for_result(self):
        while True:
            if not self._results.empty():
                return self._results.get_nowait()
            await asyncio.sleep(1)

    def kill(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            if not self._queue.empty():
                job = self._queue.get_nowait()
                result = job.target(*job.args, **job.kwargs)
                self._results.put_nowait(result)
            time.sleep(0.1)


class WorkerPool:

    def __init__(self, size):
        self._workers = []
        for i in range(size):
            worker = Worker(name=f'Worker-{i}')
            worker.start()
            self._workers.append(worker)

    @property
    def size(self):
        return len(self._workers)

    async def dispatch(self, jobs):
        busy_workers = []
        workers = iter(self._workers)
        for job in jobs:
            worker = None
            try:
                worker = next(workers)
            except StopIteration:
                workers = iter(self._workers)
                worker = next(workers)
            finally:
                worker.enqueue(job)
                busy_workers.append(worker)

        results = []
        for worker in busy_workers:
            result = await worker.wait_for_result()
            results.append(result)

        return results

    def dispose(self):
        for worker in self._workers:
            worker.kill()
