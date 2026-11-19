# Evaluating LLM Search Providers

## Context

### Current State

At time of running this evaluation (Nov 2026) I have deployed a self-hosted
LibreChat instance and integrated Tavily. I have had 27 conversations so far
which included at least one call to the `web_search` tool.

### Background

There are many search providers and adjacent solutions that can be used by LLMs
to find information to solve a problem or answer a user question:

- Exa
- Brave Search
- Perplexity
- Tavily
- Jina
- DuckDuckGo
- Startpage
- Ecosia
- Firecrawl
- Kagi
- SearXNG

A sample for how LibreChat stores messages with `web_seach` tool calls can be
found at [./example-message.json]. I have not verified if this example is
canonical, so other instances of `web_search` tool calls may look different.

Bifrost logs all my requests, but there seems to be no session ID collected
for traces sent by LibreChat.

## Goal

I want to find out, whether for my usage patterns

- the available search providers come up with significantly different search
  results,
- the differences between search results and formatting impact the quality of
  the agent's outcome, and
- how the search providers rank in different quality metrics (to be specified)
  such as
  - result relevance
  - context size (more information in less context is better)

Based on the results I may want to switch search providers, but other criteria
also factor in this decision (data privacy, regional processing within the EU,
...).

## Methodology

### Run the evaluation

From the repository root:

```sh
./evaluations/search_providers/run.sh
```

The runner downloads fresh snapshots and prepares scenarios. It stops there
until the provider experiment, baseline generation, and scoring are implemented.
From another directory, invoke the script by its path.

Set these environment variables before a fresh run:

- `EVAL_SSH_HOST`: hostname or SSH alias for the host running both services.
- `EVAL_SSH_USER`: remote SSH login name on that host.
- `BIFROST_CONTAINER`: name of the running Bifrost container.
- `BIFROST_DB`: absolute path to the live `logs.db` inside the Bifrost container.
- `LIBRECHAT_MONGO_CONTAINER`: name of the running LibreChat MongoDB container.
- `LIBRECHAT_DB`: the LibreChat database name.

Install `uv`, `ssh`, `tar`, and `sqlite3` locally. The remote SSH user needs Bash and permission
to run `docker exec` on both containers.
The Bifrost container needs `sh` and `tar`. The runner copies the live database
and any WAL, shared-memory, or rollback-journal files, then uses local SQLite
to produce a standalone snapshot. Bifrost keeps running: concurrent writes
can make the copy inconsistent or omit recent rows. Allow local disk space
for both the raw copy and the standalone snapshot during download.
The MongoDB container needs `sh` and `mongodump`, and
must expose its existing `MONGO_INITDB_ROOT_USERNAME` and
`MONGO_INITDB_ROOT_PASSWORD` environment variables to `docker exec`.
Only `messages` and `conversations` are exported.

Each invocation writes to `evaluations/search_providers/data/<UTC-timestamp>/`.
The backups are in `snapshots/` and the scenario JSONL is in `scratch/`.
Future provider requests will run afresh and write their intermediate artifacts
to `scratch/`; only final scores and reports will go in `results/`.

Pass `--reuse-inputs` to skip SSH and prepare a new dataset from the last
successfully prepared source pair. This doesn't require the SSH or container
settings. A reuse run links to the original `snapshots/`, so keep that
directory while other runs reference it.

### Prepare Evaluation dataset

To prepare an existing combined archive without downloading fresh snapshots,
run from the repository root:

```sh
uv run --group search-providers python evaluations/search_providers/prepare_dataset.py \
  --db "$BIFROST_SNAPSHOT" --librechat "$LIBRECHAT_ARCHIVE" \
  --out evaluations/search_providers/data/manual/scratch/search_provider_scenarios.jsonl
```

For separate gzipped `mongodump --archive` exports, use
`--librechat-collections MESSAGES_ARCHIVE CONVERSATIONS_ARCHIVE` instead of
`--librechat`. The script writes one replay scenario per eligible search turn.

- `history` is Bifrost's model input followed by its assistant output. It
  preserves system messages, any context compaction, and the original call batch.
- `conversationId` and `assistantMessageId` identify the LibreChat turn.
  Polars joins on exact call IDs. A turn is retained only if all its
  `web_search` calls resolve unambiguously to the same assistant message.
  The script reports unmatched, partial, and ambiguous turns separately.
- `originalSearchResults` contains LibreChat's tool output for each call in
  Bifrost call order. An unavailable output is `null`.
- `originalConversation` is the recorded LibreChat branch, rendered as
  OpenAI-style user, assistant, and tool messages. The script uses the saved
  last-response pointer only if its branch contains the matched message. It
  otherwise uses the sole compatible branch, or keeps only the known prefix
  if the ending is ambiguous.
- `originalAnswerStatus` says whether the branch ends in a complete answer.
  `originalConversationHasIncompleteMessages` also flags earlier unfinished
  or errored messages, even if the branch later ends in a complete answer.

The dataset represents retained conversations that can be traced to both
sources. Unmatched turns may have been deleted from LibreChat, but the export
does not establish why they are absent.

### Run experiment across providers

Add the `web_search` tool result to the conversation with results from across
all providers to be evaluated. Let the conversation continue until the bot
provides an answer. To account for model output flakiness, we let the
conversation continue 10 times.

### Define baseline

For each scenario, based on all the search results, generate baseline
evaluation using a top-tier LLM. This baseline evaluation dataset should
contain

- answer evaluation criteria, e.g.

```json
{
  "id": "C1",
  "requirement": "States that Paul inherits Jamis's household",
  "necessity": "essential",
}
```

It will also generate a baseline answer.

### Evaluate

Measurements:

- estimated cost for the web search based on a formula provided as input
  (different providers have different pricing)
- Token count of web search results per tool call
- Let a decision model decide on all the baseline criteria
- Let a decision model decide whether the final answer was significantly worse,
  slightly worse, same, slightly better or significantly better compared to the
  baseline answer.
- Extract facts from search results and for each decide on relevance to the
  conversation so far.

Based on these measurements we define a rubric score to get to a final
evaluation score for a provider.
