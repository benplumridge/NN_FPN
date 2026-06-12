import math
import os


def _as_plain_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return (
            f"{type(value).__name__}(shape={tuple(value.shape)}, dtype={value.dtype})"
        )
    if isinstance(value, (list, tuple)):
        return [_as_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_plain_value(item) for key, item in value.items()}
    return str(value)


def _run_name(cfg, params, context):
    template = cfg.get("name_template", "{dimension}-{ansatz}-N{N}-run{replicate}")
    fields = {**params, **context}
    try:
        return template.format(**fields)
    except KeyError:
        return None


def init_wandb(params):
    cfg = params.get("wandb", {}) or {}
    if not cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled, but wandb is not installed. Run: uv pip install wandb"
        ) from exc

    context = cfg.get("context", {}) or {}
    mode = cfg.get("mode") or os.environ.get("WANDB_MODE") or "online"
    project = cfg.get("project", "NN_FPN")
    tags = list(cfg.get("tags", []))
    tags.extend(
        str(tag) for tag in [context.get("dimension"), context.get("ansatz")] if tag
    )
    if params.get("N") is not None:
        tags.append(f"N{params['N']}")

    run_config = {
        "params": _as_plain_value(params),
        "context": _as_plain_value(context),
    }

    return wandb.init(
        project=project,
        entity=cfg.get("entity") or None,
        group=cfg.get("group") or None,
        job_type=cfg.get("job_type", "train"),
        name=_run_name(cfg, params, context),
        tags=tags,
        mode=mode,
        config=run_config,
        reinit=True,
    )


def _log_interval(params):
    cfg = params.get("wandb", {}) or {}
    return int(cfg.get("log_interval", params.get("log_interval", 1)))


def should_log_metrics(params, step, final=False):
    if final:
        return True
    interval = _log_interval(params)
    return interval > 0 and step % interval == 0


def log_metrics(run, metrics, step, params, final=False):
    if run is None:
        return
    if not should_log_metrics(params, step, final=final):
        return
    run.log(metrics, step=step)


def finish_run(run):
    if run is not None:
        run.finish()
