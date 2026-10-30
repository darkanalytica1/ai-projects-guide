"""Run the agent on every task in tests/tasks.json and check the files it produced.

  AUTO_APPROVE=1 python run_tests.py
"""
import asyncio
import json
import pathlib
import re
import shutil

import agent

tasks = json.loads(pathlib.Path("tests/tasks.json").read_text())
passed = 0
for i, t in enumerate(tasks, 1):
    shutil.rmtree(agent.WORKSPACE, ignore_errors=True)
    agent.WORKSPACE.mkdir()
    print(f"\n=== task {i}: {t['task'][:70]}")
    try:
        asyncio.run(agent.main(t["task"]))
    except Exception as e:  # budget exceeded, API error, ...
        print("run failed:", e)
    ok = True
    for c in t["checks"]:
        f = agent.WORKSPACE / c["file"]
        if not f.exists() or ("regex" in c and not re.search(c["regex"], f.read_text(), re.I)):
            ok = False
            print("  check failed:", c)
    passed += ok
    print("  PASS" if ok else "  FAIL")
print(f"\ntask pass rate: {passed}/{len(tasks)} = {passed / len(tasks):.0%}")
