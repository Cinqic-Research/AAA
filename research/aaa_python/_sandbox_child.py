"""Sandbox child for the ``aaa.python.v0`` oracle. Runs as its own process.

Started as ``python -I -S _sandbox_child.py`` with an empty environment in a
fresh temporary directory. It reads one JSON request from stdin, applies its
own resource limits *before* touching the program, compiles or executes the
program with restricted builtins, and writes one JSON result line to stdout.

It imports only the standard library and nothing from this repository, so it
behaves the same from a checkout and from an installed wheel.
"""

import builtins
import io
import json
import sys
import time


class OutputLimit(BaseException):
    """Raised inside the program when its output exceeds the declared bound."""


class CappedWriter(io.TextIOBase):
    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.size = 0
        self.truncated = False

    def writable(self):
        return True

    def write(self, text):
        if self.size + len(text) > self.limit:
            self.parts.append(text[: max(0, self.limit - self.size)])
            self.size = self.limit
            self.truncated = True
            raise OutputLimit()
        self.parts.append(text)
        self.size += len(text)
        return len(text)

    def value(self):
        return "".join(self.parts)


def apply_limits(request):
    applied: dict[str, int] = {}
    try:
        import resource
    except ImportError:  # not POSIX: recorded, never silently claimed
        return applied
    wanted = {
        "RLIMIT_CPU": request["cpu_seconds"],
        "RLIMIT_AS": request["address_space_bytes"],
        "RLIMIT_FSIZE": 0,
        "RLIMIT_NPROC": 0,
    }
    for name, value in wanted.items():
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            _, hard = resource.getrlimit(limit)
            if hard != resource.RLIM_INFINITY and hard < value:
                value = hard
            # CPU: a one-second-higher hard limit lets SIGXCPU (classified as
            # cpu_limit) arrive before the kernel's SIGKILL.
            ceiling = (
                value + 1
                if name == "RLIMIT_CPU" and (hard == resource.RLIM_INFINITY or hard > value)
                else value
            )
            resource.setrlimit(limit, (value, ceiling))
            applied[name] = value
        except (ValueError, OSError):
            pass
    return applied


def error_line(error):
    line = None
    trace = error.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code.co_filename == "<aaa-program>":
            line = trace.tb_lineno
        trace = trace.tb_next
    return line


def main():
    request = json.loads(sys.stdin.read())
    limits = apply_limits(request)
    result = {"limits_applied": limits}
    source = request["source"]
    started = time.perf_counter()
    try:
        code = compile(source, "<aaa-program>", "exec")
    except SyntaxError as error:
        result.update(status="syntax_error", exception=type(error).__name__, line=error.lineno, stdout="")
        code = None
    if code is not None and request["mode"] == "compile":
        result.update(status="valid", exception=None, line=None, stdout="")
    elif code is not None:
        allowed = {name: getattr(builtins, name) for name in request["builtins"]}
        namespace = {"__builtins__": allowed, "__name__": "__aaa_program__"}
        writer = CappedWriter(request["max_stdout_chars"])
        real_stdout = sys.stdout
        sys.stdout = writer
        try:
            exec(code, namespace)
            status, exception, line = "ok", None, None
        except OutputLimit:
            status, exception, line = "output_limit", None, None
        except BaseException as error:  # the program's own failure is the observation
            status, exception, line = "exception", type(error).__name__, error_line(error)
        finally:
            sys.stdout = real_stdout
        result.update(status=status, exception=exception, line=line, stdout=writer.value())
        result["truncated"] = writer.truncated
    result["seconds"] = time.perf_counter() - started
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
