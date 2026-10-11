"""Nén kết quả một lần chạy thành run.zip.   python -m tools.pack_outputs --run-dir outputs/yolo/<run_id>"""
import argparse

from ts_alert.runtime.pack import pack_run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    print(f"Đã nén: {pack_run_dir(args.run_dir, args.out)}")


if __name__ == "__main__":
    main()
