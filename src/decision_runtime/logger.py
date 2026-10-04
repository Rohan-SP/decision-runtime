import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class DecisionLogger:
    """Store self-contained records of workflow decisions."""

    def __init__(
        self,
        logs_dir: str | Path | None = None,
    ) -> None:
        if logs_dir is None:
            logs_dir = (
                Path(__file__).resolve().parents[1]
                / "decision_logs"
            )

        self.logs_dir = Path(logs_dir).expanduser().resolve()
        self.run_id: str | None = None
        self.run_dir: Path | None = None

    def create_run(self, run_dir: str | Path | None = None) -> dict:
        """Create a folder for one workflow run."""

        run_id = uuid4().hex
        if run_dir is None:
            run_dir = self.logs_dir / run_id

        run_dir = Path(run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=False)

        self.run_id = run_id
        self.run_dir = run_dir

        return {
            "run_id": run_id,
            "run_dir": run_dir,
        }

    def save_prefix(
        self,
        prefix_text: str,
    ) -> str:
        """Save an exact prefix snapshot and return its content ID."""

        content = prefix_text.encode("utf-8")
        prefix_id = hashlib.sha256(content).hexdigest()

        folder = self.run_dir / "prefixes"
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / f"{prefix_id}.txt"
        if not path.exists():
            path.write_bytes(content)

        return prefix_id

    def save_suffix(
        self,
        suffix_text: str,
    ) -> str:
        """Save an exact suffix snapshot and return its content ID."""

        content = suffix_text.encode("utf-8")
        suffix_id = hashlib.sha256(content).hexdigest()

        folder = self.run_dir / "suffixes"
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / f"{suffix_id}.txt"
        if not path.exists():
            path.write_bytes(content)

        return suffix_id

    def save_state(
        self,
        prefix_id: str,
        suffix_id: str,
    ) -> str:
        """Record a prefix/suffix combination and return its state ID."""

        components = {
            "prefix_id": prefix_id,
            "suffix_id": suffix_id,
        }

        identity = json.dumps(
            components,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        state_id = hashlib.sha256(identity).hexdigest()

        folder = self.run_dir / "states"
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / f"{state_id}.json"
        if not path.exists():
            record = {
                "state_id": state_id,
                **components,
            }
            path.write_text(
                json.dumps(record, indent=2),
                encoding="utf-8",
            )

        return state_id

    def save_decision(
        self,
        state_id: str,
        question: dict,
        answer: dict,
        model_info: dict,
    ) -> str:
        """Save a completed decision with its question and model information."""

        decision_id = uuid4().hex

        record = {
            "decision_id": decision_id,
            "run_id": self.run_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "state_id": state_id,
            "question": question,
            "answer": answer,
            "model_info": model_info,
        }

        folder = self.run_dir / "decisions"
        folder.mkdir(parents=True, exist_ok=True)

        path = folder / f"{decision_id}.json"
        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return decision_id

    def log_decision(
        self,
        prefix_text: str,
        suffix_text: str,
        question: dict,
        question_text: str,
        answer: dict,
        model_info: dict,
    ) -> str:
        """Save a completed decision and its exact context."""

        prefix_id = self.save_prefix(prefix_text)
        suffix_id = self.save_suffix(suffix_text)
        state_id = self.save_state(prefix_id, suffix_id)

        return self.save_decision(
            state_id=state_id,
            question={
                **question,
                "rendered_text": question_text,
            },
            answer=answer,
            model_info=model_info,
        )
