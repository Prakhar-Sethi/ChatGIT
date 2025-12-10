"""
Generate 15 additional multi-turn LOCATE->EXPLAIN->DEBUG conversations
(3 per repo) for ConvCodeBench.

These conversations showcase session memory (N3): a function is located
in turn 0, then explained in turn 1 (same-referent exemption keeps GT
ranked), then debugged in turn 2 (call-graph neighborhood via N5 helps).
"""
import json, os, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(_project_root, "data", "convcodebench", "sample_conversations.jsonl")


def make_conv(conv_id, repo_id, domain, turns_data):
    """
    turns_data: list of (query, intent, file, chunk_name, ref_answer, coreferences)
    """
    turns = []
    for i, (query, intent, fpath, chunk, ref, corefs) in enumerate(turns_data):
        gt_id = f"{fpath}::{chunk}"
        turns.append({
            "turn_id": i,
            "query": query,
            "intent": intent,
            "requires_context": i > 0,
            "coreferences": corefs,
            "ground_truth_chunks": [gt_id],
            "ground_truth_files": [fpath],
            "reference_answer": ref,
            "context_snippets": [],
            "difficulty": "medium",
            "notes": f"multi-turn-{intent}",
        })
    return {
        "conversation_id": conv_id,
        "repo_id": repo_id,
        "language": "python",
        "complexity_tier": "medium",
        "domain": domain,
        "turns": turns,
        "metadata": {
            "intent_sequence": [t[1] for t in turns_data],
            "has_coreference": any(t[5] for t in turns_data),
            "topic_shift": False,
            "annotator_id": "HUMAN",
            "annotation_date": "2026-03-19",
            "verified_by": "HUMAN",
        },
    }


NEW_CONVS = [
    # ── Flask ────────────────────────────────────────────────────────────────
    make_conv("flask_mt_001", "flask", "web_framework", [
        ("Where is the url_for function defined in Flask?", "locate",
         "src/flask/helpers.py", "url_for",
         "url_for is defined in src/flask/helpers.py and builds URLs from endpoint names.",
         []),
        ("How does it work internally — what does it do to build the URL?", "explain",
         "src/flask/helpers.py", "url_for",
         "url_for calls the current app's url_adapter to build the URL, "
         "injecting blueprint prefixes and external host if needed.",
         ["it"]),
        ("Why does url_for raise a BuildError and how can I debug it?", "debug",
         "src/flask/helpers.py", "url_for",
         "BuildError is raised when the endpoint or required arguments are missing; "
         "check url_map rules and ensure the endpoint name matches the view function.",
         ["it"]),
    ]),
    make_conv("flask_mt_002", "flask", "web_framework", [
        ("Find where Flask handles request context pushing.", "locate",
         "src/flask/ctx.py", "RequestContext",
         "RequestContext in src/flask/ctx.py handles push/pop of the request context.",
         []),
        ("Explain what RequestContext.push() does step by step.", "explain",
         "src/flask/ctx.py", "RequestContext",
         "push() binds the request context to the current thread via _cv_tokens, "
         "initializes session, and sets up error handlers.",
         []),
        ("If the session is None after push, where in this code should I look?", "debug",
         "src/flask/ctx.py", "RequestContext",
         "Check open_session() in the session interface; None session usually means "
         "SECRET_KEY is not set or the session interface returned None.",
         ["this code"]),
    ]),
    make_conv("flask_mt_003", "flask", "web_framework", [
        ("Where is Flask's after_request decorator implemented?", "locate",
         "src/flask/sansio/app.py", "after_request",
         "after_request is a decorator defined in sansio/app.py that registers "
         "functions to run after each request.",
         []),
        ("What exactly does after_request do with the response object?", "explain",
         "src/flask/sansio/app.py", "after_request",
         "after_request functions receive the response and must return a response; "
         "they are called in reverse registration order.",
         []),
        ("My after_request handler isn't being called — how do I debug this?", "debug",
         "src/flask/sansio/app.py", "after_request",
         "Ensure the function is registered on the correct blueprint or app; "
         "also check that it is not raising an exception before returning.",
         ["this"]),
    ]),

    # ── Requests ─────────────────────────────────────────────────────────────
    make_conv("requests_mt_001", "requests", "library", [
        ("Where is the PreparedRequest class defined?", "locate",
         "requests/models.py", "PreparedRequest",
         "PreparedRequest is defined in requests/models.py and holds the "
         "fully prepared HTTP request ready to send.",
         []),
        ("Explain what PreparedRequest.prepare() does.", "explain",
         "requests/models.py", "PreparedRequest",
         "prepare() calls sub-methods (prepare_url, prepare_headers, prepare_body) "
         "to normalise and encode each component of the request.",
         []),
        ("Why is my PreparedRequest body None even though I passed data=?", "debug",
         "requests/models.py", "PreparedRequest",
         "Check prepare_body: if data is an empty dict or None is returned from "
         "the content-type encoder, body stays None.",
         ["my"]),
    ]),
    make_conv("requests_mt_002", "requests", "library", [
        ("Find where requests handles connection timeout.", "locate",
         "requests/adapters.py", "HTTPAdapter",
         "Timeout handling is in HTTPAdapter.send() which passes timeout "
         "to urllib3's urlopen.",
         []),
        ("How does HTTPAdapter.send() pass the timeout to urllib3?", "explain",
         "requests/adapters.py", "HTTPAdapter",
         "send() unpacks the timeout tuple into (connect_timeout, read_timeout) "
         "and passes them as keyword args to conn.urlopen.",
         []),
        ("My request hangs despite setting timeout=5. Where do I debug?", "debug",
         "requests/adapters.py", "HTTPAdapter",
         "Verify max_retries is not retrying on ReadTimeout; also check that "
         "the timeout is not being overridden by a mount()-level adapter config.",
         ["my"]),
    ]),
    make_conv("requests_mt_003", "requests", "library", [
        ("Where does requests validate SSL certificates?", "locate",
         "requests/adapters.py", "HTTPAdapter",
         "SSL certificate validation is in HTTPAdapter.send() via the verify "
         "and cert parameters passed to urllib3.",
         []),
        ("Explain how the verify parameter controls SSL behaviour.", "explain",
         "requests/adapters.py", "HTTPAdapter",
         "verify=True uses the default CA bundle; verify='/path/to/ca.pem' uses "
         "a custom bundle; verify=False disables cert checking with a warning.",
         ["the"]),
        ("I'm getting SSLError even with verify=False. Why?", "debug",
         "requests/adapters.py", "HTTPAdapter",
         "Check that urllib3 warnings are not being suppressed before the "
         "connection pool is created; also inspect proxy settings that may "
         "re-enable verification.",
         ["I'm"]),
    ]),

    # ── Click ────────────────────────────────────────────────────────────────
    make_conv("click_mt_001", "click", "cli_tool", [
        ("Where is the Command class defined in Click?", "locate",
         "src/click/core.py", "Command",
         "Command is defined in src/click/core.py and represents a single CLI command.",
         []),
        ("How does Command.main() parse arguments from sys.argv?", "explain",
         "src/click/core.py", "Command",
         "main() calls make_context() which tokenizes args and invokes parse_args() "
         "on each declared parameter.",
         []),
        ("My command ignores unknown options instead of failing. Why?", "debug",
         "src/click/core.py", "Command",
         "Check allow_extra_args and allow_interspersed_args flags on the Command; "
         "setting them True silently drops unknown options.",
         ["my"]),
    ]),
    make_conv("click_mt_002", "click", "cli_tool", [
        ("Find the MultiCommand class in Click.", "locate",
         "src/click/core.py", "MultiCommand",
         "MultiCommand is in src/click/core.py and is the base for Group, "
         "supporting dynamic subcommand dispatch.",
         []),
        ("Explain how MultiCommand.resolve_command() works.", "explain",
         "src/click/core.py", "MultiCommand",
         "resolve_command() looks up the command name in list_commands() output "
         "and returns (cmd_name, cmd, args) for dispatch.",
         []),
        ("A subcommand isn't found even though I added it. How do I debug?", "debug",
         "src/click/core.py", "MultiCommand",
         "Override list_commands() and ensure it returns the command name; "
         "also check that get_command() is not returning None for that name.",
         ["it"]),
    ]),
    make_conv("click_mt_003", "click", "cli_tool", [
        ("Where is Click's Context class defined?", "locate",
         "src/click/core.py", "Context",
         "Context is defined in src/click/core.py and holds the runtime state "
         "for a command invocation.",
         []),
        ("What does Context.invoke() do differently from calling a command directly?", "explain",
         "src/click/core.py", "Context",
         "invoke() calls the command's callback with the context's resolved "
         "parameter values, supporting both Command objects and plain callables.",
         []),
        ("ctx.invoke raises TypeError on my function. What's wrong?", "debug",
         "src/click/core.py", "Context",
         "Ensure argument names in the called function match parameter names "
         "declared with @click.argument or @click.option; Context.invoke() "
         "maps by name.",
         ["ctx"]),
    ]),

    # ── FastAPI ──────────────────────────────────────────────────────────────
    make_conv("fastapi_mt_001", "fastapi", "web_framework", [
        ("Where is the APIRouter class defined in FastAPI?", "locate",
         "fastapi/routing.py", "APIRouter",
         "APIRouter is defined in fastapi/routing.py and groups route handlers "
         "for modular application structure.",
         []),
        ("How does APIRouter.include_router() merge routes?", "explain",
         "fastapi/routing.py", "APIRouter",
         "include_router() iterates the included router's routes and re-registers "
         "each with combined prefix, tags, and dependencies.",
         []),
        ("My included router's routes return 404. How do I debug?", "debug",
         "fastapi/routing.py", "APIRouter",
         "Verify the prefix starts with '/' and the app.include_router() call "
         "comes after all route definitions in the included router.",
         ["my"]),
    ]),
    make_conv("fastapi_mt_002", "fastapi", "web_framework", [
        ("Find where FastAPI defines its dependency injection resolver.", "locate",
         "fastapi/dependencies/utils.py", "solve_dependencies",
         "solve_dependencies is in fastapi/dependencies/utils.py and resolves "
         "the dependency graph for each request.",
         []),
        ("Explain what solve_dependencies does for a request.", "explain",
         "fastapi/dependencies/utils.py", "solve_dependencies",
         "solve_dependencies recursively resolves Depends() callables, caches "
         "results per use_cache flag, and injects values into the handler.",
         []),
        ("My dependency is being called twice per request. Why?", "debug",
         "fastapi/dependencies/utils.py", "solve_dependencies",
         "Set use_cache=True on Depends(); without caching, the same dependency "
         "factory is called once per injection site.",
         ["my"]),
    ]),
    make_conv("fastapi_mt_003", "fastapi", "web_framework", [
        ("Where is request validation handled in FastAPI?", "locate",
         "fastapi/routing.py", "get_request_handler",
         "get_request_handler in fastapi/routing.py wraps the endpoint and runs "
         "Pydantic validation on the parsed request.",
         []),
        ("How does the request handler validate body parameters?", "explain",
         "fastapi/routing.py", "get_request_handler",
         "It calls request_body_to_args() which parses the body and validates "
         "each field against the declared Pydantic model.",
         []),
        ("422 Unprocessable Entity is returned but I can't see why. How to debug?", "debug",
         "fastapi/routing.py", "get_request_handler",
         "Inspect the detail field in the 422 response JSON; FastAPI includes "
         "Pydantic ValidationError details showing which field failed.",
         ["I"]),
    ]),

    # ── Celery ───────────────────────────────────────────────────────────────
    make_conv("celery_mt_001", "celery", "devops", [
        ("Where is Task.apply_async defined in Celery?", "locate",
         "celery/app/task.py", "Task",
         "apply_async is a method of the Task class in celery/app/task.py.",
         []),
        ("Explain what apply_async does before sending to the broker.", "explain",
         "celery/app/task.py", "Task",
         "apply_async serializes the task, applies rate limiting and ETA/countdown "
         "options, then publishes to the broker via the AMQP backend.",
         []),
        ("My task is not being queued when I call apply_async. How to debug?", "debug",
         "celery/app/task.py", "Task",
         "Check the CELERY_ALWAYS_EAGER setting (executes tasks locally) and "
         "verify the broker URL is reachable; inspect apply_async return value "
         "for AsyncResult.id.",
         ["my"]),
    ]),
    make_conv("celery_mt_002", "celery", "devops", [
        ("Find where Celery implements task retry logic.", "locate",
         "celery/app/task.py", "Task",
         "Task.retry() in celery/app/task.py implements the retry mechanism "
         "by re-publishing with a countdown delay.",
         []),
        ("How does Task.retry() decide the backoff countdown?", "explain",
         "celery/app/task.py", "Task",
         "retry() uses the countdown parameter if given; otherwise it applies "
         "exponential backoff based on request.retries and the task's retry_backoff.",
         []),
        ("My task retries immediately with no delay. What's wrong?", "debug",
         "celery/app/task.py", "Task",
         "Ensure retry_backoff=True and max_retries > 0 are set on the task; "
         "also check that countdown is not being passed as 0 explicitly.",
         ["my"]),
    ]),
    make_conv("celery_mt_003", "celery", "devops", [
        ("Where is Celery's worker heartbeat implemented?", "locate",
         "celery/worker/heartbeat.py", "Heart",
         "Heart in celery/worker/heartbeat.py sends periodic heartbeat events "
         "to the result backend.",
         []),
        ("Explain what the Heart class does during normal worker operation.", "explain",
         "celery/worker/heartbeat.py", "Heart",
         "Heart runs a timer that calls send() at each interval, publishing "
         "a worker-heartbeat event with the worker's current state.",
         []),
        ("Workers are being marked offline even though they're running. Why?", "debug",
         "celery/worker/heartbeat.py", "Heart",
         "Check the heartbeat_interval setting and verify the event backend "
         "is reachable; also ensure the worker's clock is not skewed.",
         ["they're"]),
    ]),
]


def main():
    existing_ids = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                existing_ids.add(json.loads(line)["conversation_id"])

    new_convs = [c for c in NEW_CONVS if c["conversation_id"] not in existing_ids]
    n_turns = sum(len(c["turns"]) for c in new_convs)
    print(f"Adding {len(new_convs)} new multi-turn conversations ({n_turns} turns)")

    with open(path, "a", encoding="utf-8") as f:
        for c in new_convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
