"""PyInstaller entry point for the InterviewOS desktop sidecar."""

from multiprocessing import freeze_support

from interview_os.desktop.sidecar import main

if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
