#!/usr/bin/env bash
set -euo pipefail
umask 077

reuse=false
limit=25
while (( $# )); do
    case "$1" in
        --reuse-inputs) reuse=true ;;
        --limit)
            [[ ${2-} =~ ^[1-9][0-9]*$ ]] || {
                echo '--limit requires a positive integer' >&2
                exit 2
            }
            limit=$2
            shift ;;
        *) echo "Usage: $0 [--reuse-inputs] [--limit N]" >&2; exit 2 ;;
    esac
    shift
done

evaluation=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data="$evaluation/data"
mkdir -p "$data"
run_dir=$(mktemp -d "$data/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
scratch="$run_dir/scratch"
snapshots="$run_dir/snapshots"
mkdir -- "$scratch"

remote() {
    local command
    printf -v command ' %q' "$@"
    ssh -T -l "$EVAL_SSH_USER" -- "$EVAL_SSH_HOST" "bash -s --$command"
}

download_snapshots() {
    : "${EVAL_SSH_HOST:?Set EVAL_SSH_HOST}"
    : "${EVAL_SSH_USER:?Set EVAL_SSH_USER}"
    : "${BIFROST_CONTAINER:?Set BIFROST_CONTAINER}"
    : "${BIFROST_DB:?Set BIFROST_DB}"
    : "${LIBRECHAT_MONGO_CONTAINER:?Set LIBRECHAT_MONGO_CONTAINER}"
    : "${LIBRECHAT_DB:?Set LIBRECHAT_DB}"

    mkdir -- "$scratch/downloads"
    echo 'Downloading Bifrost snapshot'
    mkdir -- "$scratch/bifrost"
    remote "$BIFROST_CONTAINER" "$BIFROST_DB" <<'REMOTE' | tar -xf - -C "$scratch/bifrost"
set -eu
docker exec -- "$1" sh -eu -c '
    cd "$(dirname "$1")"
    db=${1##*/}
    set -- "./$db"
    for suffix in -wal -shm -journal; do
        if [ -f "$db$suffix" ]; then set -- "$@" "./$db$suffix"; fi
    done
    exec tar -cf - "$@"
' sh "$2"
REMOTE
    (
        cd "$scratch/bifrost"
        # SQLite rebuilds the WAL index locally from the copied WAL.
        rm -f -- "${BIFROST_DB##*/}-shm"
        sqlite3 -batch -bail "./${BIFROST_DB##*/}" '.backup ../downloads/logs.db'
    )
    rm -r -- "$scratch/bifrost"

    echo 'Downloading LibreChat snapshots'
    local collection
    for collection in messages conversations; do
        remote "$LIBRECHAT_MONGO_CONTAINER" "$LIBRECHAT_DB" "$collection" \
            > "$scratch/downloads/$collection.gz" <<'REMOTE'
set -eu
docker exec -- "$1" sh -eu -c '
    exec mongodump --quiet --gzip --archive --db "$1" --collection "$2" \
        --authenticationDatabase admin \
        --username "${MONGO_INITDB_ROOT_USERNAME:?}" \
        --password "${MONGO_INITDB_ROOT_PASSWORD:?}"
' sh "$2" "$3"
REMOTE
    done
    mv -- "$scratch/downloads" "$snapshots"
}

if $reuse; then
    if [[ -f "$data/latest-snapshots" ]] &&
        IFS= read -r previous < "$data/latest-snapshots" &&
        [[ -s "$data/$previous/snapshots/logs.db" &&
           -s "$data/$previous/snapshots/messages.gz" &&
           -s "$data/$previous/snapshots/conversations.gz" ]]; then
        ln -s -- "../$previous/snapshots" "$snapshots"
    else
        echo 'No reusable snapshots. Run without --reuse-inputs first.' >&2
        exit 1
    fi
else
    download_snapshots
fi

dataset="$scratch/search_provider_scenarios.jsonl"
TMPDIR="$scratch" uv run --project "$evaluation/../.." --group search-providers \
    python "$evaluation/prepare_dataset.py" --db "$snapshots/logs.db" \
    --librechat-collections "$snapshots/messages.gz" "$snapshots/conversations.gz" \
    --limit "$limit" --out "$dataset"
if ! $reuse; then
    printf '%s\n' "${run_dir##*/}" > "$scratch/latest-snapshots"
    mv -f -- "$scratch/latest-snapshots" "$data/latest-snapshots"
fi
printf 'Dataset prepared: %s (%s scenarios)\n' "$dataset" "$(wc -l < "$dataset")"
