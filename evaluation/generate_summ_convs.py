"""Generate 25 SUMMARIZE conversations (5 per repo) for the benchmark."""
import json, os, sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(_project_root, "data", "convcodebench", "sample_conversations.jsonl")


def make_summ(conv_id, repo_id, domain, class_name, file_path, query, answer, difficulty="medium"):
    gt_id = f"{file_path}::{class_name}"
    return {
        "conversation_id": conv_id,
        "repo_id": repo_id,
        "language": "python",
        "complexity_tier": "medium",
        "domain": domain,
        "turns": [{
            "turn_id": 0,
            "query": query,
            "intent": "summarize",
            "requires_context": False,
            "coreferences": [],
            "ground_truth_chunks": [gt_id],
            "ground_truth_files": [file_path],
            "reference_answer": answer,
            "context_snippets": [],
            "difficulty": difficulty,
            "notes": "summarize-class",
        }],
        "metadata": {
            "intent_sequence": ["summarize"],
            "has_coreference": False,
            "topic_shift": False,
            "annotator_id": "HUMAN",
            "annotation_date": "2026-03-19",
            "verified_by": "HUMAN",
        },
    }


SUMM_CONVS = [
    # Flask
    make_summ("flask_summ_001", "flask", "web_framework", "Flask", "src/flask/app.py",
              "Give me an overview of the main Flask application class and what it is responsible for.",
              "Flask is the core WSGI application object wiring together routing, request handling, configuration, and extension support."),
    make_summ("flask_summ_002", "flask", "web_framework", "Blueprint", "src/flask/blueprints.py",
              "Summarize the Blueprint class and how it enables modular Flask applications.",
              "Blueprint groups related views and defers registration until mounted on an app via register_blueprint."),
    make_summ("flask_summ_003", "flask", "web_framework", "SecureCookieSession", "src/flask/sessions.py",
              "What is the SecureCookieSession class and what session data does it manage?",
              "SecureCookieSession is a dict-like session backed by a signed cookie, tracking modification and access flags."),
    make_summ("flask_summ_004", "flask", "web_framework", "Request", "src/flask/wrappers.py",
              "What does the Flask Request class add on top of Werkzeug base request?",
              "Flask Request extends Werkzeug Request to add JSON parsing, routing_exception, and blueprint url_rule support."),
    make_summ("flask_summ_005", "flask", "web_framework", "BlueprintSetupState", "src/flask/sansio/blueprints.py",
              "What is BlueprintSetupState and when is it used during blueprint registration?",
              "BlueprintSetupState holds transient state during blueprint registration and is passed to deferred setup functions.",
              difficulty="hard"),
    # Requests
    make_summ("requests_summ_001", "requests", "library", "Session", "requests/sessions.py",
              "Give me an architectural overview of the Session class in the requests library.",
              "Session persists cookies, auth, headers, and proxies across requests and implements connection pooling via HTTPAdapter."),
    make_summ("requests_summ_002", "requests", "library", "RequestsCookieJar", "requests/cookies.py",
              "What is RequestsCookieJar and how does it differ from a plain dict?",
              "RequestsCookieJar is a MutableMapping backed by a cookielib CookieJar supporting domain/path-aware cookie lookup."),
    make_summ("requests_summ_003", "requests", "library", "Request", "requests/models.py",
              "Summarize the Request model class and what data it holds before being prepared.",
              "Request is a user-facing data class holding method, url, headers, files, data, params, auth, cookies, and hooks before preparation."),
    make_summ("requests_summ_004", "requests", "library", "SessionRedirectMixin", "requests/sessions.py",
              "What is the role of SessionRedirectMixin in requests and what redirect behavior does it encapsulate?",
              "SessionRedirectMixin provides resolve_redirects and helpers for rebuilding auth, method, and headers across redirect chains."),
    make_summ("requests_summ_005", "requests", "library", "HTTPAdapter", "requests/adapters.py",
              "How does the HTTPAdapter class work and what is its role in the requests transport layer?",
              "HTTPAdapter implements the transport layer using urllib3 connection pools, handling SSL, proxies, and retries."),
    # Click
    make_summ("click_summ_001", "click", "cli_tool", "Option", "src/click/core.py",
              "Summarize the Option class in Click and what it represents.",
              "Option represents a named CLI option supporting multiple values, prompting, defaults, and required flags."),
    make_summ("click_summ_002", "click", "cli_tool", "Argument", "src/click/core.py",
              "What does the Argument class do in Click and how does it differ from Option?",
              "Argument represents positional CLI parameters; unlike Option it has no name prefix and is required by default."),
    make_summ("click_summ_003", "click", "cli_tool", "Group", "src/click/core.py",
              "Give an architectural overview of the Group class and how it manages sub-commands.",
              "Group is a MultiCommand collecting named sub-commands and dispatching based on the first CLI argument."),
    make_summ("click_summ_004", "click", "cli_tool", "BaseCommand", "src/click/core.py",
              "What is BaseCommand in Click and what interface does it define for all commands?",
              "BaseCommand is the abstract base for all Click commands defining make_context, invoke, and main."),
    make_summ("click_summ_005", "click", "cli_tool", "HelpFormatter", "src/click/formatting.py",
              "What is HelpFormatter responsible for in Click help text rendering?",
              "HelpFormatter manages terminal width, indentation, and section writing for Click --help output."),
    # FastAPI
    make_summ("fastapi_summ_001", "fastapi", "web_framework", "FastAPI", "fastapi/applications.py",
              "Give me an overview of the FastAPI application class and what it adds over Starlette.",
              "FastAPI extends Starlette with automatic OpenAPI schema generation, dependency injection, and Pydantic validation."),
    make_summ("fastapi_summ_002", "fastapi", "web_framework", "BackgroundTasks", "fastapi/background.py",
              "Summarize the BackgroundTasks class and how FastAPI handles deferred work.",
              "BackgroundTasks queues callables to run after response is sent, useful for non-blocking post-request work."),
    make_summ("fastapi_summ_003", "fastapi", "web_framework", "Dependant", "fastapi/dependencies/models.py",
              "What is the Dependant model class in FastAPI and what does it track?",
              "Dependant is a dataclass modeling a dependency graph node, tracking path/query/header/body params and sub-dependencies.",
              difficulty="hard"),
    make_summ("fastapi_summ_004", "fastapi", "web_framework", "OAuth2PasswordRequestFormStrict",
              "fastapi/security/oauth2.py",
              "What is OAuth2PasswordRequestFormStrict and how does it differ from the non-strict form?",
              "OAuth2PasswordRequestFormStrict requires grant_type=password, rejecting other grant types unlike the non-strict version.",
              difficulty="hard"),
    make_summ("fastapi_summ_005", "fastapi", "web_framework", "Depends", "fastapi/params.py",
              "How does the Depends class work in FastAPI dependency injection?",
              "Depends is a sentinel class marking a callable as a dependency; FastAPI resolves it via solve_dependencies."),
    # Celery
    make_summ("celery_summ_001", "celery", "devops", "Task", "celery/app/task.py",
              "Give me an architectural overview of the Celery Task base class and what it is responsible for.",
              "Task is the base class for all Celery tasks providing apply_async, delay, retry, and request-context lifecycle."),
    make_summ("celery_summ_002", "celery", "devops", "MaxRetriesExceededError", "celery/exceptions.py",
              "What is MaxRetriesExceededError in Celery and when is it raised?",
              "MaxRetriesExceededError is raised by Task.retry when the max_retries limit has been reached.",
              difficulty="easy"),
    make_summ("celery_summ_003", "celery", "devops", "AMQP", "celery/app/amqp.py",
              "Summarize the AMQP class in Celery and what messaging responsibilities it owns.",
              "AMQP manages Kombu-based message production, queue/exchange declaration, and serialization for Celery task dispatch."),
    make_summ("celery_summ_004", "celery", "devops", "crontab", "celery/schedules.py",
              "What is the crontab schedule class in Celery and how does it specify periodic task timing?",
              "crontab is a schedule class using Unix cron patterns to determine when periodic tasks should run."),
    make_summ("celery_summ_005", "celery", "devops", "Request", "celery/worker/request.py",
              "What is the role of the Celery worker Request class in task execution lifecycle?",
              "The worker Request tracks in-flight task state including task id, time start, retries, and worker hostname.",
              difficulty="hard"),
]


def main():
    existing_ids = set()
    with open(path) as f:
        for line in f:
            if line.strip():
                existing_ids.add(json.loads(line)["conversation_id"])

    new_convs = [c for c in SUMM_CONVS if c["conversation_id"] not in existing_ids]
    print(f"Adding {len(new_convs)} new SUMMARIZE conversations "
          f"({sum(len(c['turns']) for c in new_convs)} SUMMARIZE turns)")

    with open(path, "a") as f:
        for c in new_convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
