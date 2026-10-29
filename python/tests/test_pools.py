import os
import shutil
import subprocess
import sys
import textwrap

import pytest

# Fails one pthread_create on request; preloaded, it comes before glibc's.
SHIM = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>

static int starts_left = -1;

void fail_after(int n) { __atomic_store_n(&starts_left, n, __ATOMIC_SEQ_CST); }

int pthread_create(pthread_t* t, const pthread_attr_t* a, void* (*f)(void*), void* arg) {
    static int (*real)(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
    if (!real) *(void**)&real = dlsym(RTLD_NEXT, "pthread_create");
    int left = __atomic_load_n(&starts_left, __ATOMIC_SEQ_CST);
    while (left >= 0 && !__atomic_compare_exchange_n(&starts_left, &left, left - 1, 0,
                                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
    if (left == 0) return EAGAIN;
    return real(t, a, f, arg);
}
"""

CHILD = textwrap.dedent(
    """
    import ctypes, sys
    import bounded_compositions as bc

    shim = ctypes.CDLL(sys.argv[1])
    s = bc.Space(20, [0] * 5, [10] * 5)
    shim.fail_after(1)  # the pool's second thread fails to start
    try:
        bc.iid_sequence(s, 1).points(0, 5000, workers=4)
        sys.exit("no error")
    except RuntimeError:
        pass
    shim.fail_after(-1)
    x = bc.iid_sequence(s, 1).points(0, 5000, workers=4)
    assert (x == bc.iid_sequence(s, 1).points(0, 5000)).all()
    """
)


@pytest.mark.skipif(sys.platform != "linux", reason="interposes glibc's pthread_create")
def test_a_pool_that_fails_to_start_raises_and_the_next_call_works(tmp_path):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler for the shim")
    src, lib = tmp_path / "shim.c", tmp_path / "shim.so"
    src.write_text(SHIM)
    subprocess.run([cc, "-shared", "-fPIC", "-o", str(lib), str(src), "-ldl"], check=True)
    env = {**os.environ, "LD_PRELOAD": str(lib)}
    r = subprocess.run(
        [sys.executable, "-c", CHILD, str(lib)], env=env, capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 0, r.stderr
