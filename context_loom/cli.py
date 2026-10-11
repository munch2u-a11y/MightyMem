from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .demo import render_demo_markdown, run_demo


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="context-loom")
    subcommands = result.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="run the four-pulse receipt-backed Maple demonstration")
    demo.add_argument("--database", default=":memory:")
    demo.add_argument("--workspace", default=None)
    demo.add_argument("--json", action="store_true")

    gestate_cmd = subcommands.add_parser("gestate", help="gestate and pair a Context Loom agent mind with Granite 4.2 8B")
    gestate_cmd.add_argument("--directory", default="states/granite42", help="directory for agent state and databases")
    gestate_cmd.add_argument("--workspace", default="workspaces/granite42", help="directory for workspace files")
    gestate_cmd.add_argument("--human-name", default="User")
    gestate_cmd.add_argument("--agent-name", default="LoomAgent")
    gestate_cmd.add_argument("--taste-schema", default="builder", choices=["balanced", "curious", "deliberate", "builder"])
    gestate_cmd.add_argument("--model", default="granite4.2:8b")
    gestate_cmd.add_argument("--embed-model", default="qwen3-embedding:0.6b")
    gestate_cmd.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    gestate_cmd.add_argument("--scope", default="default")
    gestate_cmd.add_argument("--timeout", type=float, default=None, help="per-call timeout; default is None (uncapped)")
    gestate_cmd.add_argument("--temperature", type=float, default=0.25)
    gestate_cmd.add_argument(
        "--context-tokens", type=int, default=None,
        help="physical model context; default requests the installed model's native context",
    )
    gestate_cmd.add_argument(
        "--kv-cache-type", choices=["f16", "q8_0", "q4_0"], default="f16",
        help="Ollama server KV format; quantized formats require verified server PID",
    )
    gestate_cmd.add_argument("--ollama-server-pid", type=int, default=None)
    gestate_cmd.add_argument(
        "--private-thinking-policy",
        choices=["adaptive", "always", "never"],
        default="adaptive",
        help="adaptive reasons for action stages and relative-date resolution",
    )

    chat_cmd = subcommands.add_parser("chat", help="interactive prompt loop with a gestated Loom agent")
    chat_cmd.add_argument("--directory", default="states/granite42")
    chat_cmd.add_argument("--workspace", default="workspaces/granite42")
    chat_cmd.add_argument("--model", default="granite4.2:8b")
    chat_cmd.add_argument("--embed-model", default="qwen3-embedding:0.6b")
    chat_cmd.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    chat_cmd.add_argument("--scope", default="default")
    chat_cmd.add_argument("--human-name", default="User")
    chat_cmd.add_argument("--agent-name", default="LoomAgent")
    chat_cmd.add_argument("--taste-schema", default="builder", choices=["balanced", "curious", "deliberate", "builder"])
    chat_cmd.add_argument(
        "--context-tokens", type=int, default=None,
        help="physical model context; default requests the installed model's native context",
    )
    chat_cmd.add_argument(
        "--kv-cache-type", choices=["f16", "q8_0", "q4_0"], default="f16",
        help="Ollama server KV format; quantized formats require verified server PID",
    )
    chat_cmd.add_argument("--ollama-server-pid", type=int, default=None)
    chat_cmd.add_argument("--temperature", type=float, default=0.25)
    chat_cmd.add_argument(
        "--private-thinking-policy",
        choices=["adaptive", "always", "never"],
        default="adaptive",
        help="adaptive reasons for action stages and relative-date resolution",
    )

    doctor = subcommands.add_parser(
        "doctor", help="write a read-only readiness receipt for the composed local product"
    )
    doctor.add_argument("--model", default="granite4.2:8b")
    doctor.add_argument("--embed-model", default="qwen3-embedding:0.6b")
    doctor.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    doctor.add_argument(
        "--context-tokens", type=int, default=None,
        help="physical model context; default requests the installed model's native context",
    )
    doctor.add_argument(
        "--kv-cache-type", choices=["f16", "q8_0", "q4_0"], default="f16",
        help="Ollama server KV format; quantized formats require verified server PID",
    )
    doctor.add_argument("--ollama-server-pid", type=int, default=None)
    doctor.add_argument("--temperature", type=float, default=0.25)
    doctor.add_argument(
        "--timeout", type=float, default=None,
        help="model metadata request timeout; default is None (uncapped)",
    )
    doctor.add_argument("--output", type=Path, default=None)

    locomo = subcommands.add_parser(
        "locomo-retrieval-diagnostic",
        help="measure evidence-ID retrieval for one LoCoMo conversation; no answer model or product score",
    )
    locomo.add_argument("--dataset", default=os.environ.get("LOCOMO_DATASET", "locomo10.json"))
    locomo.add_argument("--conversation-index", type=int, default=1)
    locomo.add_argument("--embed-model", default="qwen3-embedding:0.6b")
    locomo.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    locomo.add_argument(
        "--kv-cache-type", choices=["f16", "q8_0", "q4_0"], default="f16",
        help="verified Ollama server KV format",
    )
    locomo.add_argument(
        "--ollama-server-pid", type=int, required=True,
        help="PID of the local-only Ollama server used for embeddings",
    )
    locomo.add_argument("--output-dir", default="artifacts/locomo-retrieval-diagnostic")
    locomo.add_argument("--timeout", type=float, default=None,
                    help="per-call timeout; default is none, let it finish")
    locomo.add_argument("--compiler-context-tokens", type=int, default=1500)
    locomo.add_argument("--compiler-reserved-output-tokens", type=int, default=400)

    channel_cmd = subcommands.add_parser(
        "channel", help="run multi-channel communications router for MightyMem"
    )
    channel_cmd.add_argument(
        "--adapter", choices=["telegram", "discord", "piper-voice"], default="telegram",
        help="main communications channel adapter type (telegram, discord, or piper-voice)",
    )
    channel_cmd.add_argument(
        "--telegram-bot-token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        help="Telegram Bot API token",
    )
    channel_cmd.add_argument(
        "--discord-bot-token", default=os.environ.get("DISCORD_BOT_TOKEN", ""),
        help="Discord Bot Token",
    )
    channel_cmd.add_argument(
        "--discord-channel-id", default=os.environ.get("DISCORD_CHANNEL_ID", ""),
        help="Discord Channel ID for bot polling & message routing",
    )
    channel_cmd.add_argument(
        "--piper-model-path", default=os.environ.get("PIPER_MODEL_PATH", "piper_en_US.onnx"),
        help="Path to local Piper TTS ONNX model file",
    )
    channel_cmd.add_argument("--db", default="mightymem_channels.sqlite", help="user profile store SQLite database path")
    channel_cmd.add_argument("--loom-db", default="mightymem_store.sqlite", help="ContextLoom store SQLite database path")

    credentials_cmd = subcommands.add_parser(
        "credentials", help="safely manage API keys and credentials in ~/.mightymem/.env"
    )
    credentials_cmd.add_argument("action", nargs="?", choices=["list", "set", "path"], default="list")
    credentials_cmd.add_argument("key", nargs="?", default=None, help="Credential key name (e.g., TELEGRAM_BOT_TOKEN)")
    credentials_cmd.add_argument("value", nargs="?", default=None, help="Credential secret value to store")

    daemon_cmd = subcommands.add_parser(
        "daemon", help="run 24/7 autonomous background consciousness daemon for MightyMem"
    )
    daemon_cmd.add_argument("--db", default="mightymem_daemon.sqlite", help="Path to persistent SQLite database")
    daemon_cmd.add_argument("--scope", default="default", help="Memory scope partition")
    daemon_cmd.add_argument("--model", default="granite4.1:8b", help="Local LLM model name via Ollama")
    daemon_cmd.add_argument("--embed-model", default="qwen3-embedding:0.6b", help="Local embedding model name via Ollama")
    daemon_cmd.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama API base URL")
    return result


def main(argv: list[str] | None = None) -> int:
    from .credentials import get_credential, get_default_credentials_dir, load_credentials, save_credential
    load_credentials()

    args = parser().parse_args(argv)

    if args.command == "demo":
        result = run_demo(args.database, args.workspace)
        print(json.dumps(result, indent=2) if args.json else render_demo_markdown(result))
        return 0
    if args.command == "gestate":
        from .gestation import GestationConfig, gestate_loom_agent

        config = GestationConfig(
            human_name=args.human_name,
            agent_name=args.agent_name,
            taste_schema=args.taste_schema,
            model_name=args.model,
            embedding_model=args.embed_model,
            ollama_url=args.ollama_url,
            scope=args.scope,
            timeout=args.timeout,
            maximum_tokens=None,  # uncapped
            temperature=args.temperature,
            context_tokens=args.context_tokens,
            kv_cache_type=args.kv_cache_type,
            ollama_server_pid=args.ollama_server_pid,
            private_thinking_policy=args.private_thinking_policy,
        )
        receipt = gestate_loom_agent(
            directory=args.directory,
            workspace=args.workspace,
            config=config,
        )
        print(json.dumps(receipt, indent=2))
        return 0
    if args.command == "chat":
        from .habitus_agent import HabitusLoomAgent
        from .deployment import verify_ollama_server_runtime
        from habitus_ai.agent_memory import LocalModels

        models = LocalModels(
            chat=args.model,
            space_id=args.embed_model,
            embedding=args.embed_model,
            num_gpu=-1,
            think=True,
            base_url=args.ollama_url,
            context_tokens=args.context_tokens,
            kv_cache_type=args.kv_cache_type,
            ollama_server_pid=args.ollama_server_pid,
        )
        if args.kv_cache_type != "f16" or args.ollama_server_pid is not None:
            verify_ollama_server_runtime(
                models,
                server_pid=args.ollama_server_pid,
                kv_cache_type=args.kv_cache_type,
            )

        agent = HabitusLoomAgent(
            directory=args.directory,
            workspace=args.workspace,
            scope=args.scope,
            human_name=args.human_name,
            agent_name=args.agent_name,
            taste_schema=args.taste_schema,
            models=models,
        )
        print(f"MightyMem agent gestated. Interacting as {args.human_name} -> {args.agent_name} [{args.scope}]. Type 'exit' to quit.")
        try:
            while True:
                user_input = input(f"{args.human_name}> ").strip()
                if not user_input or user_input.lower() in ("exit", "quit"):
                    break
                reply = agent.handle(user_input, scope=args.scope)
                print(f"{args.agent_name}> {reply}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nEnding chat session.")
        return 0
    if args.command == "doctor":
        from .doctor import run_deployment_doctor, write_doctor_receipt

        receipt = run_deployment_doctor(
            chat_model=args.model,
            embedding_model=args.embed_model,
            ollama_url=args.ollama_url,
            context_tokens=args.context_tokens,
            kv_cache_type=args.kv_cache_type,
            ollama_server_pid=args.ollama_server_pid,
        )
        if args.output:
            write_doctor_receipt(args.output, receipt)
            print(f"wrote doctor receipt to {args.output}")
        else:
            print(json.dumps(receipt, indent=2))
        return 0
    if args.command == "locomo-retrieval-diagnostic":
        from .locomo_evaluation import render_retrieval_report, run_locomo_retrieval

        common = {
            "dataset_path": args.dataset,
            "conversation_index": args.conversation_index,
            "embed_model": args.embed_model,
            "ollama_url": args.ollama_url,
            "kv_cache_type": args.kv_cache_type,
            "ollama_server_pid": args.ollama_server_pid,
            "timeout": args.timeout,
            "compiler_context_tokens": args.compiler_context_tokens,
            "compiler_reserved_output_tokens": args.compiler_reserved_output_tokens,
        }
        result = run_locomo_retrieval(args.output_dir, **common)
        print(render_retrieval_report(result))
        print(f"wrote {Path(args.output_dir) / 'REPORT.md'}")
        return 0
    if args.command == "channel":
        import asyncio
        from .engine import ContextLoom
        from .channels import ChannelRouter, DiscordChannelAdapter, PiperVoiceChannelAdapter, TelegramChannelAdapter, UserProfileStore

        if args.adapter == "telegram" and not args.telegram_bot_token:
            print("Error: --telegram-bot-token or TELEGRAM_BOT_TOKEN environment variable required.")
            return 1
        if args.adapter == "discord" and not args.discord_bot_token:
            print("Error: --discord-bot-token or DISCORD_BOT_TOKEN environment variable required.")
            return 1

        loom = ContextLoom(Path(args.loom_db))
        profiles = UserProfileStore(Path(args.db))
        
        # Instantiate HabitusLoomAgent runner if available
        def agent_runner(text: str, scope: str, profile: Any) -> str:
            try:
                from .habitus_agent import HabitusLoomAgent
                agent = HabitusLoomAgent(loom=loom)
                return agent.handle(text, scope=scope)
            except Exception as err:
                logger.debug("HabitusLoomAgent fallback to router synthesizer: %s", err)
                return router._synthesize_response(text, scope, profile)

        router = ChannelRouter(loom=loom, profiles=profiles, agent_runner=agent_runner)


        if args.adapter == "telegram":
            adapter = TelegramChannelAdapter(bot_token=args.telegram_bot_token)
            router.register_adapter(adapter)
        elif args.adapter == "discord":
            adapter = DiscordChannelAdapter(bot_token=args.discord_bot_token, channel_id_target=args.discord_channel_id)
            router.register_adapter(adapter)
        elif args.adapter == "piper-voice":
            adapter = PiperVoiceChannelAdapter(piper_model_path=args.piper_model_path)
            router.register_adapter(adapter)

        async def _run():
            print(f"Starting MightyMem Channel Router ({args.adapter}) ... Press Ctrl+C to stop.")
            await router.start_all()
            try:
                while True:
                    await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                await router.stop_all()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            pass
        return 0
    if args.command == "credentials":
        if args.action == "path":
            env_file = get_default_credentials_dir() / ".env"
            print(f"User credentials file path: {env_file}")
            return 0
        if args.action == "set":
            if not args.key:
                print("Error: `context-loom credentials set KEY [VALUE]` requires a KEY name.")
                return 1
            key = args.key.upper()
            val = args.value
            if not val:
                import getpass
                val = getpass.getpass(f"Enter {key} (typing hidden): ")
            if not val.strip():
                print("Error: No credential value provided.")
                return 1
            saved_path = save_credential(key, val.strip())
            print(f"Saved {key} securely to {saved_path} (chmod 0600).")
            return 0

        if args.action == "list":
            env_file = get_default_credentials_dir() / ".env"
            print(f"=== MightyMem Credentials ({env_file}) ===")
            keys = ["TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID", "PIPER_MODEL_PATH"]
            for k in keys:
                val = get_credential(k)
                if val:
                    masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "***"
                    print(f"  {k}: {masked}")
                else:
                    print(f"  {k}: [Not Configured]")
            return 0

    if args.command == "daemon":
        from scripts.mightymem_daemon import main as daemon_main
        sys.argv = [sys.argv[0], "--db", args.db, "--scope", args.scope, "--model", args.model, "--embed-model", args.embed_model, "--ollama-url", args.ollama_url]
        return daemon_main()

    raise RuntimeError(f"unhandled command: {args.command}")



if __name__ == "__main__":
    raise SystemExit(main())
