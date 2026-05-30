"""Run the strict LSL verification suite."""
import subprocess
import sys


def run(*args):
    print(f"\n>>> {' '.join(args)}")
    result = subprocess.run([sys.executable, *args], text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    print("LSL strict verification")
    run("test_lsl.py")
    run("benchmark_sdr_phase1.py")
    run("benchmark_pc_phase2.py")
    run("benchmark_cortical_column_sequence.py")
    run("test_extreme_strict_infra.py")
    run("benchmark_goal_strict.py", "--profile", "smoke")
    run("benchmark_goal_strict.py")
    print("\nAll strict checks passed.")


if __name__ == "__main__":
    main()
