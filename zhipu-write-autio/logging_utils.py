"""
日志工具：将 stdout/stderr 同步写入日志文件
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Union


class _Tee(io.TextIOBase):
    """简单的 Tee 流，将写入内容同步到多个目标"""

    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        for stream in self.streams:
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def __getattr__(self, name: str):
        return getattr(self.streams[0], name)


@contextmanager
def capture_run_logs(
    prefix: str = "run",
    log_dir: Optional[Union[str, Path]] = None,
) -> Iterator[Path]:
    """
    将 stdout/stderr 同步写入日志文件，返回日志文件路径
    """

    base_dir = Path(log_dir) if log_dir else Path(__file__).parent / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = base_dir / f"{prefix}_{timestamp}.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        tee_stdout = _Tee(original_stdout, log_file)
        tee_stderr = _Tee(original_stderr, log_file)

        sys.stdout = tee_stdout
        sys.stderr = tee_stderr

        try:
            print(f"📝 日志输出: {log_path}")
            yield log_path
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


