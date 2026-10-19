"""Runs every test file in this folder and reports the PASS/FAIL totals. Run: python tests/run_all.py"""
import glob, os, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
failed = 0
for path in sorted(glob.glob(os.path.join(here, 'test_*.py'))):
    result = subprocess.run([sys.executable, '-X', 'utf8', path], capture_output=True, text=True, encoding='utf-8')
    lines = result.stdout.splitlines()
    passes, fails = [l for l in lines if l.startswith('PASS')], [l for l in lines if l.startswith('FAIL')]
    crashed = result.returncode != 0
    failed += len(fails) + crashed
    print(f"{os.path.basename(path):28s} {len(passes):3d} passed, {len(fails)} failed" + (" (CRASHED)" if crashed else ""))
    for line in fails:
        print("   ", line)
    if crashed:
        print("   ", result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no error output")
print("ALL PASSED" if failed == 0 else f"{failed} FAILURE(S)")
sys.exit(1 if failed else 0)
