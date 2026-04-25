"""CLI model switching for the OpenAI-compatible provider."""

from __future__ import annotations

from agent.core.provider import resolve_provider_config, save_provider_config


def print_model_listing(config, console) -> None:
    provider = resolve_provider_config(config)
    console.print("[bold]Current model:[/bold]")
    console.print(f"  {config.model_name}")
    if provider:
        console.print(f"\n[bold]Provider:[/bold]\n  {provider.base_url}")
    console.print(
        "\n[dim]Use /model <model-name> to change the model for the current "
        "OpenAI-compatible provider.\nUse /provider setup to edit base URL, "
        "API key, model, and context window.[/dim]"
    )


def switch_model(model_id: str, config, session, console) -> None:
    if not model_id:
        console.print("[bold red]Missing model name.[/bold red]")
        return
    provider = resolve_provider_config(config)
    config.model_name = model_id
    if provider:
        provider.model = model_id
        save_provider_config(provider)
    if session is not None:
        session.update_model(model_id)
    console.print(f"[green]Model switched to {model_id}[/green]")
