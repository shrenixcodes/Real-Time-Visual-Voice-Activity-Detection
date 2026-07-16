"""Start the Shrenix browser dashboard for Visual VAD and local STT."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from visual_vad.dashboard import DashboardConfig, create_dashboard_server


def parse_audio_device(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Shrenix Visual VAD dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--no-stt", action="store_true", help="Run the dashboard without local transcription")
    parser.add_argument("--stt-backend", choices=("whisper", "vosk"), default="whisper")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--whisper-cache", type=Path, default=Path("models/whisper"))
    parser.add_argument("--language", default="en")
    parser.add_argument("--audio-device", type=parse_audio_device, default=None)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument(
        "--live-whisper-model",
        default=None,
        help="Fast model for live drafts (default: tiny.en for English, tiny otherwise)",
    )
    parser.add_argument(
        "--live-update-seconds",
        type=float,
        default=0.25,
        help="Cadence for revisable live Whisper transcript updates",
    )
    parser.add_argument(
        "--live-window-seconds",
        type=float,
        default=2.0,
        help="Recent-audio window used for revisable live transcript updates",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the dashboard URL")
    args = parser.parse_args()

    config = DashboardConfig(
        camera=args.camera,
        width=args.width,
        height=args.height,
        max_faces=args.max_faces,
        enable_stt=not args.no_stt,
        stt_backend=args.stt_backend,
        whisper_model=args.whisper_model,
        whisper_cache=args.whisper_cache,
        language=args.language or None,
        audio_device=args.audio_device,
        initial_prompt=args.initial_prompt,
        live_model_size=args.live_whisper_model,
        live_update_seconds=args.live_update_seconds,
        live_window_seconds=args.live_window_seconds,
    )
    server = create_dashboard_server(args.host, args.port, config)
    url = f"http://{args.host}:{args.port}"
    print(f"Shrenix dashboard: {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.runtime.stop()
        server.server_close()


if __name__ == "__main__":
    main()
