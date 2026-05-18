from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm as atqdm

try:
    import aiohttp
    import torch
    from transformers import AutoConfig, AutoTokenizer
except ImportError:
    pass

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

def canonical_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_file_hash(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def manifest_path_for(output_path: str) -> Path:
    return Path(f"{output_path}.meta.json")


@dataclass
class Item:
    uid: str
    sample_index: int
    messages: List[Dict[str, Any]]


def read_jsonl(path: str, n: int) -> List[Item]:
    data: List[Item] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("uid")
                messages = obj.get("prompt")
                if not uid or not isinstance(messages, list):
                    raise ValueError("missing uid or prompt")
                for sample_index in range(n):
                    data.append(
                        Item(uid=uid, sample_index=sample_index, messages=messages)
                    )
            except Exception as exc:
                print(f"[WARN] skip bad line {line_number}: {exc}", file=sys.stderr)
    return data


def load_done_pairs(output_path: str) -> set[tuple[str, int]]:
    done_pairs: set[tuple[str, int]] = set()
    if not os.path.exists(output_path):
        return done_pairs

    with open(output_path, "r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = obj.get("uid")
            if not isinstance(uid, str):
                raise ValueError(
                    f"invalid uid in {output_path}:{line_number}: {obj.get('uid')!r}"
                )
            sample_index = obj.get("sample_index", 0)
            if not isinstance(sample_index, int):
                raise ValueError(
                    f"invalid sample_index in {output_path}:{line_number}: "
                    f"{sample_index!r}"
                )
            pair = (uid, sample_index)
            if pair in done_pairs:
                raise ValueError(
                    f"duplicate uid/sample_index pair in {output_path}:{line_number}: "
                    f"{pair}"
                )
            done_pairs.add(pair)
    return done_pairs


class JsonlWriter:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "a", encoding="utf-8")
        self._lock = asyncio.Lock()

    async def write_line(self, obj: Dict[str, Any]):
        serialized = json.dumps(obj, ensure_ascii=False)
        async with self._lock:
            self._fp.write(serialized + "\n")
            self._fp.flush()
            os.fsync(self._fp.fileno())

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass


class RateLimiter:
    def __init__(self, rate: int = 0, per: float = 60.0):
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        if self.rate <= 0:
            return
        async with self.lock:
            while self.tokens <= 0:
                now = time.monotonic()
                elapsed = now - self.updated
                refill = int(elapsed * (self.rate / self.per))
                if refill > 0:
                    self.tokens = min(self.rate, self.tokens + refill)
                    self.updated = now
                if self.tokens <= 0:
                    await asyncio.sleep(max(0.05, self.per / self.rate / 2))
            self.tokens -= 1


class WorkerPool:
    def __init__(
        self,
        client: Optional[AsyncOpenAI],
        model: str,
        out: JsonlWriter,
        concurrency: int,
        limiter: RateLimiter,
        args: argparse.Namespace,
        progress_bar=None,
        api_mode: str = "openai",
        raw_url: str | None = None,
        tokenizer=None,
        model_config=None,
        run_config_hash: str = "",
    ):
        self.client = client
        self.model = model
        self.out = out
        self.sem = asyncio.Semaphore(concurrency)
        self.limiter = limiter
        self.args = args
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.concurrency = concurrency
        self.shutdown = False
        self.progress_bar = progress_bar
        self.api_mode = api_mode
        self.raw_url = raw_url
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.aio_session = None
        self.run_config_hash = run_config_hash

    async def run_items(self, items: List[Item], done_pairs: set[tuple[str, int]]):
        if self.api_mode == "raw":
            timeout = aiohttp.ClientTimeout(total=self.args.request_timeout)
            connector = aiohttp.TCPConnector(limit=self.concurrency)
            self.aio_session = aiohttp.ClientSession(timeout=timeout, connector=connector)

        tasks = []
        for item in items:
            if (item.uid, item.sample_index) in done_pairs:
                continue
            tasks.append(asyncio.create_task(self._with_semaphore(item)))

        if not tasks:
            print("[INFO] nothing to do — all UID/sample_index pairs already present")
            return

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            print("[WARN] Cancelled; awaiting in-flight tasks to finish...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if self.aio_session:
                await self.aio_session.close()

    async def _with_semaphore(self, item: Item):
        async with self.sem:
            await self._process_item(item)

    async def _process_item(self, item: Item):
        if self.shutdown:
            return

        await self.limiter.acquire()

        start_time = time.perf_counter()
        attempt = 0
        max_attempts = 6
        last_error: Optional[str] = None
        response_content = ""
        meta_info = {}
        success = False

        while attempt < max_attempts:
            attempt += 1
            try:
                if self.api_mode == "raw":
                    loop = asyncio.get_running_loop()
                    input_ids = await loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            item.messages,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        ),
                    )
                    payload = {
                        "input_ids": input_ids,
                        "sampling_params": {
                            "temperature": self.temperature,
                            "max_new_tokens": self.max_tokens,
                            "top_k": self.top_k,
                            "top_p": self.top_p,
                        },
                        "return_routed_experts": self.args.enable_routed_experts_stats,
                    }
                    async with self.aio_session.post(self.raw_url, json=payload) as resp:
                        resp.raise_for_status()
                        res_json = await resp.json()
                        output_ids = res_json.get("output_ids", [])
                        response_content = await loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.decode(
                                output_ids, skip_special_tokens=False
                            ),
                        )
                        meta_info = res_json["meta_info"].copy()

                        if (
                            self.args.enable_routed_experts_stats
                            and not self.args.save_routed_experts
                        ):
                            meta_info.pop("routed_experts", None)

                        if self.args.enable_routed_experts_stats:
                            routed_experts_tensor = torch.tensor(
                                res_json["meta_info"]["routed_experts"]
                            )
                            num_normal_experts = (
                                getattr(self.model_config, "num_experts", None)
                                or getattr(
                                    self.model_config, "n_routed_experts", None
                                )
                                if self.model_config is not None
                                else None
                            )
                            if num_normal_experts is None:
                                raise ValueError("num_normal_experts is not set")
                            normal_expert_mask = (
                                routed_experts_tensor < num_normal_experts
                            )
                            meta_info["normal_expert_ratio"] = float(
                                normal_expert_mask.sum()
                            ) / normal_expert_mask.numel()
                else:
                    create_kwargs = dict(
                        model=self.model,
                        messages=item.messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                    )
                    if self.top_k is not None and self.top_k >= 0:
                        create_kwargs["extra_body"] = {"top_k": self.top_k}
                    resp = await self.client.chat.completions.create(**create_kwargs)
                    response_content = (
                        resp.choices[0].message.content if resp.choices else ""
                    )
                    meta_info = {
                        "id": getattr(resp, "id", None),
                        "usage": (
                            resp.usage.model_dump()
                            if hasattr(resp, "usage") and resp.usage is not None
                            else None
                        ),
                    }

                success = True
                break
            except Exception as exc:
                last_error = str(exc)
                backoff = min(20.0, 1.5**attempt)
                lowered = last_error.lower()
                if any(
                    token in lowered
                    for token in [
                        "rate",
                        "overloaded",
                        "timeout",
                        "temporarily",
                        "429",
                        "502",
                        "503",
                        "504",
                        "server disconnected",
                    ]
                ):
                    await asyncio.sleep(backoff)
                    continue
                break

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        to_write = {
            "uid": item.uid,
            "sample_index": item.sample_index,
            "ok": success,
            "model": self.model,
            "created": int(time.time()),
            "latency_ms": latency_ms,
            "messages": item.messages,
            "response": response_content if success else None,
            "error": last_error if not success else None,
            "meta": {
                **meta_info,
                "attempt": attempt,
                "n": self.args.n,
                "run_config_hash": self.run_config_hash,
            },
        }

        await self.out.write_line(to_write)
        if self.progress_bar:
            self.progress_bar.update(1)


def build_client() -> AsyncOpenAI:
    if AsyncOpenAI is None:
        raise RuntimeError(
            "This script requires the openai Python package (>=1.0).\n"
            "Install: pip install --upgrade openai aiohttp transformers"
        )
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY", "dummy")
    org = os.getenv("OPENAI_ORG")
    return AsyncOpenAI(api_key=api_key, base_url=base_url, organization=org)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent sample-level rollout manager"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rate", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--strict-resume-config", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument(
        "--api-mode",
        type=str,
        choices=["openai", "raw"],
        default="openai",
        help="Choose 'openai' for SDK or 'raw' for HTTP requests",
    )
    parser.add_argument(
        "--raw-url",
        type=str,
        default="http://127.0.0.1:25431/generate",
        help="Endpoint for raw mode",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model (required for raw mode)",
    )
    parser.add_argument(
        "--enable-routed-experts-stats",
        action="store_true",
        default=False,
        help="Enable routed experts stats",
    )
    parser.add_argument(
        "--save-routed-experts",
        action="store_true",
        default=False,
        help="Save routed experts to file",
    )
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be >= 1")
    return args


def build_run_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = {
        "input_path": os.path.abspath(args.input),
        "input_hash": compute_file_hash(args.input),
        "model": args.model,
        "api_mode": args.api_mode,
        "raw_url": args.raw_url if args.api_mode == "raw" else None,
        "model_path": os.path.abspath(args.model_path) if args.model_path else None,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    manifest["run_config_hash"] = sha256_text(canonical_json(manifest))
    return manifest


def prepare_resume_state(args: argparse.Namespace) -> tuple[Dict[str, Any], set[tuple[str, int]]]:
    output_path = Path(args.output)
    manifest_path = manifest_path_for(args.output)
    current_manifest = build_run_manifest(args)
    current_config = {
        key: value for key, value in current_manifest.items() if key != "run_config_hash"
    }

    if output_path.exists():
        if args.strict_resume_config:
            if not manifest_path.exists():
                raise ValueError(
                    f"Missing manifest for strict resume validation: {manifest_path}."
                )
            with manifest_path.open("r", encoding="utf-8") as file_obj:
                saved_manifest = json.load(file_obj)
            saved_config = {
                key: value
                for key, value in saved_manifest.items()
                if key != "run_config_hash"
            }
            if saved_config != current_config:
                raise ValueError(
                    "Resume config mismatch.\n"
                    f"saved={canonical_json(saved_config)}\n"
                    f"current={canonical_json(current_config)}"
                )
            manifest = saved_manifest
        else:
            manifest = current_manifest
        done_pairs = load_done_pairs(args.output)
        return manifest, done_pairs

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file_obj:
        json.dump(current_manifest, file_obj, ensure_ascii=False, indent=2)
    return current_manifest, set()


async def amain():
    args = parse_args()

    tokenizer = None
    model_config = None
    if args.api_mode == "raw":
        if not args.model_path:
            print("ERROR: --model-path is required for raw mode", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Loading tokenizer from {args.model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model_config = AutoConfig.from_pretrained(args.model_path)

    manifest, done_pairs = prepare_resume_state(args)
    items = read_jsonl(args.input, n=args.n)
    print(f"[INFO] loaded {len(items)} input samples from benchmark x n")
    if done_pairs:
        print(f"[INFO] resume: {len(done_pairs)} uid/sample_index pairs already done")

    out = JsonlWriter(args.output)
    client = None
    if args.api_mode == "openai":
        client = build_client().with_options(timeout=args.request_timeout)

    limiter = RateLimiter(rate=args.rate, per=60.0)
    pending = [item for item in items if (item.uid, item.sample_index) not in done_pairs]
    bar = atqdm(total=len(pending), desc="Rollout progress", unit="sample")

    pool = WorkerPool(
        client=client,
        model=args.model,
        out=out,
        concurrency=args.concurrency,
        limiter=limiter,
        args=args,
        progress_bar=bar,
        api_mode=args.api_mode,
        raw_url=args.raw_url,
        tokenizer=tokenizer,
        model_config=model_config,
        run_config_hash=manifest["run_config_hash"],
    )

    loop = asyncio.get_running_loop()

    def handle_signal(*_):
        print("\n[WARN] signal received, will stop after current tasks...")
        pool.shutdown = True

    try:
        loop.add_signal_handler(signal.SIGINT, handle_signal)
        loop.add_signal_handler(signal.SIGTERM, handle_signal)
    except NotImplementedError:
        pass

    try:
        await pool.run_items(items, done_pairs)
    finally:
        bar.close()
        out.close()


def main():
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        print("[INFO] interrupted")


if __name__ == "__main__":
    main()
