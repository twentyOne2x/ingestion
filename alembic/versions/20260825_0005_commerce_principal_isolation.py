"""Bind complete commerce lineage components to one authorization realm.

Existing components containing an ACP bridge become ``acp_internal``. Residual
pre-principal history is retained in ``system_internal`` quarantine. The
upgrade aborts on orphaned edges, malformed JSON references, or mixed explicit
ownership before RLS and future-write lineage triggers are enabled.

Revision ID: 20260825_0005
Revises: 20260825_0004
Create Date: 2026-08-25
"""

import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa

from alembic import op

revision: str = "20260825_0005"
down_revision: str | None = "20260825_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


COMMERCE_TABLES = (
    "channel_quotes",
    "quote_videos",
    "checkout_sessions",
    "channel_packs",
    "pack_batches",
    "pack_videos",
    "channel_orders",
    "payment_receipts",
    "acp_job_bridges",
    "entitlements",
)


_GRAPH_COLUMNS = {
    "channel_quotes": ("id", "request_json"),
    "quote_videos": ("id", "quote_id"),
    "checkout_sessions": ("id", "quote_ids_json", "line_items_json"),
    "channel_packs": ("id",),
    "pack_batches": ("id", "pack_id", "quote_id", "checkout_session_id"),
    "pack_videos": ("id", "pack_id", "batch_id", "quote_id"),
    "channel_orders": (
        "id",
        "quote_id",
        "checkout_session_id",
        "pack_id",
        "batch_id",
    ),
    "payment_receipts": ("id", "checkout_session_id", "order_id"),
    "acp_job_bridges": (
        "acp_job_id",
        "quote_id",
        "checkout_session_id",
        "order_id",
        "pack_id",
        "request_json",
        "delivery_json",
    ),
    "entitlements": ("id", "pack_id"),
}
_GRAPH_PRIMARY_KEYS = {
    "acp_job_bridges": "acp_job_id",
    **{
        table_name: "id"
        for table_name in COMMERCE_TABLES
        if table_name != "acp_job_bridges"
    },
}
_RUNTIME_AUTHORITIES = {"gateway", "acp_internal", "system_internal"}
_SOURCE_VIDEO_ARCHIVE_STATES_V4 = (
    "pending_discovery",
    "retained_remote_verified",
    "retained_hot_verified",
    "partial_only",
)
_SOURCE_VIDEO_ARCHIVE_STATES_V5 = (
    *_SOURCE_VIDEO_ARCHIVE_STATES_V4,
    "blocked_public_age_gate",
)


def _decoded_json(value: Any, *, label: str, expected_type: type) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label} is not valid JSON") from exc
    if not isinstance(value, expected_type):
        raise TypeError(f"{label} must be a {expected_type.__name__}")
    return value


def _node_key(table_name: str, row_id: Any) -> tuple[str, str]:
    if row_id is None or isinstance(row_id, (dict, list, tuple, set)):
        raise RuntimeError(f"{table_name} has an invalid primary key")
    normalized = str(row_id)
    if not normalized:
        raise RuntimeError(f"{table_name} has an empty primary key")
    return table_name, normalized


def _json_foreign_key(value: Any, *, label: str, optional: bool) -> str | None:
    if value is None or value == "":
        if optional:
            return None
        raise RuntimeError(f"{label} is missing its required parent")
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string id")
    return value


def _existing_signature(
    node: dict[str, Any],
) -> tuple[str, str | None, str | None] | None:
    authority = str(node["authority_kind"] or "")
    tenant_id = node["tenant_id"]
    principal_id = node["principal_user_id"]
    if authority == "legacy_internal":
        if tenant_id is not None or principal_id is not None:
            raise RuntimeError(
                "legacy commerce row unexpectedly carries gateway principals"
            )
        return None
    if authority not in _RUNTIME_AUTHORITIES:
        raise RuntimeError(f"commerce row has unsupported authority {authority!r}")
    if authority == "gateway":
        if not tenant_id or not principal_id:
            raise RuntimeError(
                "gateway commerce row is missing its exact principal pair"
            )
    elif tenant_id is not None or principal_id is not None:
        raise RuntimeError(
            "internal commerce row unexpectedly carries gateway principals"
        )
    return authority, tenant_id, principal_id


def _load_commerce_graph(
    connection,
    table_names: Sequence[str],
    *,
    include_ownership: bool,
) -> tuple[
    tuple[str, ...],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], set[tuple[str, str]]],
]:
    """Load and validate every declared relational and structured JSON edge."""
    selected_tables = tuple(
        table_name for table_name in COMMERCE_TABLES if table_name in table_names
    )
    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    rows_by_table: dict[str, list[dict[str, Any]]] = {}
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = {}

    for table_name in selected_tables:
        columns = list(_GRAPH_COLUMNS[table_name])
        if include_ownership:
            columns.extend(("authority_kind", "tenant_id", "principal_user_id"))
        rows = [
            dict(row)
            for row in connection.execute(
                sa.text(f"SELECT {', '.join(columns)} FROM {table_name}")
            ).mappings()
        ]
        rows_by_table[table_name] = rows
        primary_key = _GRAPH_PRIMARY_KEYS[table_name]
        for row in rows:
            key = _node_key(table_name, row[primary_key])
            if key in nodes:
                raise RuntimeError(f"duplicate commerce graph node {key[0]}:{key[1]}")
            nodes[key] = {**row, "raw_primary_key": row[primary_key]}
            adjacency[key] = set()

    def connect(
        source_key: tuple[str, str],
        target_table: str,
        target_id: Any,
        *,
        label: str,
        optional: bool = False,
    ) -> None:
        if target_id is None or target_id == "":
            if optional:
                return
            raise RuntimeError(f"{label} is missing its required parent")
        target_key = _node_key(target_table, target_id)
        if target_key not in nodes:
            raise RuntimeError(
                f"{label} references missing {target_table}:{target_key[1]}"
            )
        adjacency[source_key].add(target_key)
        adjacency[target_key].add(source_key)

    for table_name, rows in rows_by_table.items():
        primary_key = _GRAPH_PRIMARY_KEYS[table_name]
        for row in rows:
            source = _node_key(table_name, row[primary_key])
            label = f"{table_name}:{source[1]}"
            if table_name == "channel_quotes":
                request = _decoded_json(
                    row["request_json"],
                    label=f"{label}.request_json",
                    expected_type=dict,
                )
                connect(
                    source,
                    "channel_packs",
                    _json_foreign_key(
                        request.get("pack_id"),
                        label=f"{label}.request_json.pack_id",
                        optional=True,
                    ),
                    label=f"{label}.request_json.pack_id",
                    optional=True,
                )
            elif table_name == "quote_videos":
                connect(
                    source, "channel_quotes", row["quote_id"], label=f"{label}.quote_id"
                )
            elif table_name == "checkout_sessions":
                quote_ids = _decoded_json(
                    row["quote_ids_json"],
                    label=f"{label}.quote_ids_json",
                    expected_type=list,
                )
                normalized_quote_ids = [
                    _json_foreign_key(
                        quote_id,
                        label=f"{label}.quote_ids_json",
                        optional=False,
                    )
                    for quote_id in quote_ids
                ]
                if not normalized_quote_ids:
                    raise RuntimeError(
                        f"{label}.quote_ids_json must contain at least one quote id"
                    )
                if len(set(normalized_quote_ids)) != len(normalized_quote_ids):
                    raise RuntimeError(
                        f"{label}.quote_ids_json contains duplicate quote ids"
                    )
                for quote_id in normalized_quote_ids:
                    connect(
                        source,
                        "channel_quotes",
                        quote_id,
                        label=f"{label}.quote_ids_json",
                    )
                line_items = _decoded_json(
                    row["line_items_json"],
                    label=f"{label}.line_items_json",
                    expected_type=list,
                )
                line_item_quote_ids: list[str] = []
                for index, line_item in enumerate(line_items):
                    if not isinstance(line_item, dict):
                        raise TypeError(
                            f"{label}.line_items_json[{index}] must be a dict"
                        )
                    quote_id = _json_foreign_key(
                        line_item.get("quote_id"),
                        label=f"{label}.line_items_json[{index}].quote_id",
                        optional=False,
                    )
                    assert quote_id is not None
                    line_item_quote_ids.append(quote_id)
                    connect(
                        source,
                        "channel_quotes",
                        quote_id,
                        label=f"{label}.line_items_json[{index}].quote_id",
                    )
                if line_item_quote_ids != normalized_quote_ids:
                    raise RuntimeError(
                        f"{label}.line_items_json quote ids must exactly match "
                        "quote_ids_json"
                    )
            elif table_name in {"pack_batches", "pack_videos", "channel_orders"}:
                for column_name, target_table in (
                    ("pack_id", "channel_packs"),
                    ("quote_id", "channel_quotes"),
                    ("checkout_session_id", "checkout_sessions"),
                    ("batch_id", "pack_batches"),
                ):
                    if column_name in row:
                        connect(
                            source,
                            target_table,
                            row[column_name],
                            label=f"{label}.{column_name}",
                        )
            elif table_name == "payment_receipts":
                connect(
                    source,
                    "checkout_sessions",
                    row["checkout_session_id"],
                    label=f"{label}.checkout_session_id",
                )
                connect(
                    source,
                    "channel_orders",
                    row["order_id"],
                    label=f"{label}.order_id",
                    optional=True,
                )
            elif table_name == "acp_job_bridges":
                bridge_edges = (
                    ("quote_id", "channel_quotes"),
                    ("checkout_session_id", "checkout_sessions"),
                    ("order_id", "channel_orders"),
                    ("pack_id", "channel_packs"),
                )
                for column_name, target_table in bridge_edges:
                    connect(
                        source,
                        target_table,
                        row[column_name],
                        label=f"{label}.{column_name}",
                        optional=True,
                    )
                request = _decoded_json(
                    row["request_json"],
                    label=f"{label}.request_json",
                    expected_type=dict,
                )
                delivery = _decoded_json(
                    row["delivery_json"],
                    label=f"{label}.delivery_json",
                    expected_type=dict,
                )
                for document_name, document in (
                    ("request_json", request),
                    ("delivery_json", delivery),
                ):
                    document_job_id = _json_foreign_key(
                        document.get("acp_job_id"),
                        label=f"{label}.{document_name}.acp_job_id",
                        optional=True,
                    )
                    if document_job_id is not None and document_job_id != source[1]:
                        raise RuntimeError(
                            f"{label}.{document_name}.acp_job_id disagrees with "
                            "the bridge primary key"
                        )
                json_edges = (
                    ("request_json", request, "pack_id", "channel_packs", "pack_id"),
                    (
                        "delivery_json",
                        delivery,
                        "quote_id",
                        "channel_quotes",
                        "quote_id",
                    ),
                    (
                        "delivery_json",
                        delivery,
                        "order_id",
                        "channel_orders",
                        "order_id",
                    ),
                    ("delivery_json", delivery, "pack_id", "channel_packs", "pack_id"),
                    ("delivery_json", delivery, "batch_id", "pack_batches", None),
                )
                for (
                    document_name,
                    document,
                    key_name,
                    target_table,
                    column_name,
                ) in json_edges:
                    target_id = _json_foreign_key(
                        document.get(key_name),
                        label=f"{label}.{document_name}.{key_name}",
                        optional=True,
                    )
                    if (
                        target_id is not None
                        and column_name is not None
                        and row[column_name] not in (None, "")
                        and str(row[column_name]) != target_id
                    ):
                        raise RuntimeError(
                            f"{label}.{document_name}.{key_name} disagrees with "
                            f"{column_name}"
                        )
                    connect(
                        source,
                        target_table,
                        target_id,
                        label=f"{label}.{document_name}.{key_name}",
                        optional=True,
                    )
                if not adjacency[source]:
                    raise RuntimeError(
                        f"{label} is detached from every commerce lifecycle row"
                    )
            elif table_name == "entitlements":
                connect(
                    source, "channel_packs", row["pack_id"], label=f"{label}.pack_id"
                )

    return selected_tables, nodes, adjacency


def _commerce_components(
    nodes: dict[tuple[str, str], dict[str, Any]],
    adjacency: dict[tuple[str, str], set[tuple[str, str]]],
) -> list[list[tuple[str, str]]]:
    components: list[list[tuple[str, str]]] = []
    visited: set[tuple[str, str]] = set()
    for start in sorted(nodes):
        if start in visited:
            continue
        component: list[tuple[str, str]] = []
        pending = [start]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            pending.extend(sorted(adjacency[current] - visited, reverse=True))
        components.append(component)
    return components


def _reconcile_commerce_lineage(connection, table_names: Sequence[str]) -> None:
    """Classify complete commerce components before RLS can hide split history.

    Every edge is treated as undirected.  A component containing an ACP bridge is
    ACP-owned; a pre-existing component without a bridge is quarantined as
    system-internal.  Explicit runtime ownership is preserved only when the whole
    component has one exact signature.  Invalid JSON, missing parents, and mixed
    signatures abort before any row is updated.
    """

    selected_tables, nodes, adjacency = _load_commerce_graph(
        connection, table_names, include_ownership=True
    )

    assignments: dict[tuple[str, str], tuple[str, str | None, str | None]] = {}
    for component in _commerce_components(nodes, adjacency):
        signatures = {
            signature
            for key in component
            if (signature := _existing_signature(nodes[key])) is not None
        }
        if any(key[0] == "acp_job_bridges" for key in component):
            signatures.add(("acp_internal", None, None))
        if len(signatures) > 1:
            rendered = ", ".join(repr(value) for value in sorted(signatures))
            raise RuntimeError(
                f"commerce component rooted at {min(component)} crosses ownership: {rendered}"
            )
        signature = next(iter(signatures), ("system_internal", None, None))
        for key in component:
            assignments[key] = signature

    for left, targets in adjacency.items():
        for right in targets:
            if assignments[left] != assignments[right]:
                raise RuntimeError(
                    f"commerce edge {left} -> {right} crosses reconciled ownership"
                )

    for key in sorted(nodes):
        node = nodes[key]
        target = assignments[key]
        current = _existing_signature(node)
        if current == target:
            continue
        if current is not None:
            raise RuntimeError(
                f"commerce row {key} would require rewriting explicit ownership"
            )
        connection.execute(
            sa.text(
                f"UPDATE {key[0]} SET authority_kind=:authority_kind, "
                "tenant_id=:tenant_id, principal_user_id=:principal_user_id "
                f"WHERE {_GRAPH_PRIMARY_KEYS[key[0]]}=:row_id"
            ),
            {
                "authority_kind": target[0],
                "tenant_id": target[1],
                "principal_user_id": target[2],
                "row_id": node["raw_primary_key"],
            },
        )

    legacy_count = sum(
        int(
            connection.execute(
                sa.text(
                    f"SELECT count(*) FROM {table_name} "
                    "WHERE authority_kind = 'legacy_internal'"
                )
            ).scalar_one()
        )
        for table_name in selected_tables
    )
    if legacy_count:
        raise RuntimeError(
            f"commerce lineage reconciliation left {legacy_count} legacy rows"
        )


def _assert_downgrade_reconstructible(connection) -> None:
    """Refuse to discard ownership that a later re-upgrade cannot reconstruct."""

    _, nodes, adjacency = _load_commerce_graph(
        connection, COMMERCE_TABLES, include_ownership=True
    )
    for component in _commerce_components(nodes, adjacency):
        signatures = {
            signature
            for key in component
            if (signature := _existing_signature(nodes[key])) is not None
        }
        if len(signatures) != 1:
            raise RuntimeError(
                f"commerce component rooted at {min(component)} has ambiguous ownership"
            )
        current = next(iter(signatures))
        reconstructible = (
            ("acp_internal", None, None)
            if any(key[0] == "acp_job_bridges" for key in component)
            else ("system_internal", None, None)
        )
        if current != reconstructible:
            raise RuntimeError(
                "downgrade would erase non-reconstructible commerce ownership for "
                f"component rooted at {min(component)}: {current!r}"
            )


def _archive_state_constraint(states: Sequence[str]) -> str:
    rendered = ", ".join(repr(state) for state in states)
    return f"archive_state IN ({rendered})"


def _replace_source_video_archive_state_constraint(states: Sequence[str]) -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_constraint(
            "ck_source_videos_archive_state",
            "source_videos",
            type_="check",
        )
        op.create_check_constraint(
            "ck_source_videos_archive_state",
            "source_videos",
            _archive_state_constraint(states),
        )
        return
    with op.batch_alter_table("source_videos", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_source_videos_archive_state",
            type_="check",
        )
        batch_op.create_check_constraint(
            "ck_source_videos_archive_state",
            _archive_state_constraint(states),
        )


def _create_scheduler_quote_projection(connection, *, existing: bool) -> None:
    """Create and rebuild the tenant-free facts consumed by the scheduler."""
    table_name = "scheduler_quote_video_projection"
    if not existing:
        op.create_table(
            table_name,
            sa.Column("quote_video_id", sa.Integer(), nullable=False),
            sa.Column("video_id", sa.String(length=64), nullable=False),
            sa.Column("position", sa.Integer(), nullable=False),
            sa.Column("included", sa.Boolean(), nullable=False),
            sa.Column("status", sa.String(length=64), nullable=False),
            sa.PrimaryKeyConstraint("quote_video_id"),
        )
        op.create_index(
            "ix_scheduler_quote_video_projection_video_id",
            table_name,
            ["video_id"],
        )
        op.create_index(
            "ix_scheduler_quote_video_projection_status",
            table_name,
            ["status"],
        )
    else:
        existing_indexes = {
            str(index["name"])
            for index in sa.inspect(connection).get_indexes(table_name)
        }
        for index_name, column_name in (
            ("ix_scheduler_quote_video_projection_video_id", "video_id"),
            ("ix_scheduler_quote_video_projection_status", "status"),
        ):
            if index_name not in existing_indexes:
                op.create_index(index_name, table_name, [column_name])
        connection.execute(sa.text(f"DELETE FROM {table_name}"))

    connection.execute(
        sa.text(
            "INSERT INTO scheduler_quote_video_projection"
            "(quote_video_id,video_id,position,included,status) "
            "SELECT id,video_id,position,included,status FROM quote_videos"
        )
    )


def _create_postgres_scheduler_projection_triggers() -> None:
    op.execute(
        """
        CREATE FUNCTION public.icmfyi_sync_scheduler_quote_video_projection()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $function$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                DELETE FROM public.scheduler_quote_video_projection
                WHERE quote_video_id = OLD.id;
                RETURN OLD;
            END IF;

            INSERT INTO public.scheduler_quote_video_projection(
                quote_video_id,
                video_id,
                position,
                included,
                status
            ) VALUES (
                NEW.id,
                NEW.video_id,
                NEW.position,
                NEW.included,
                NEW.status
            )
            ON CONFLICT (quote_video_id) DO UPDATE SET
                video_id = EXCLUDED.video_id,
                position = EXCLUDED.position,
                included = EXCLUDED.included,
                status = EXCLUDED.status;
            RETURN NEW;
        END
        $function$
        """
    )
    op.execute(
        "REVOKE ALL ON FUNCTION "
        "public.icmfyi_sync_scheduler_quote_video_projection() FROM PUBLIC"
    )
    op.execute(
        """
        CREATE FUNCTION public.icmfyi_guard_scheduler_quote_video_projection()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY INVOKER
        SET search_path = pg_catalog, public
        AS $function$
        BEGIN
            IF pg_trigger_depth() <= 1 THEN
                RAISE EXCEPTION 'scheduler quote projection is trigger-maintained'
                    USING ERRCODE = '55000';
            END IF;
            IF TG_OP = 'DELETE' THEN
                RETURN OLD;
            END IF;
            RETURN NEW;
        END
        $function$
        """
    )
    op.execute(
        "REVOKE ALL ON FUNCTION "
        "public.icmfyi_guard_scheduler_quote_video_projection() FROM PUBLIC"
    )
    op.execute(
        "CREATE TRIGGER scheduler_quote_video_projection_guard "
        "BEFORE INSERT OR UPDATE OR DELETE ON scheduler_quote_video_projection "
        "FOR EACH ROW EXECUTE FUNCTION "
        "public.icmfyi_guard_scheduler_quote_video_projection()"
    )
    op.execute(
        "CREATE TRIGGER quote_videos_scheduler_projection "
        "AFTER INSERT OR UPDATE OR DELETE ON quote_videos FOR EACH ROW "
        "EXECUTE FUNCTION public.icmfyi_sync_scheduler_quote_video_projection()"
    )


def _create_postgres_constraints_and_policies() -> None:
    for table_name in COMMERCE_TABLES:
        op.alter_column(table_name, "authority_kind", nullable=False)
        op.create_check_constraint(
            f"ck_{table_name}_commerce_authority",
            table_name,
            "authority_kind IN "
            "('gateway', 'acp_internal', 'system_internal', 'legacy_internal')",
        )
        op.create_check_constraint(
            f"ck_{table_name}_commerce_owner_shape",
            table_name,
            "(authority_kind = 'gateway' AND tenant_id IS NOT NULL "
            "AND principal_user_id IS NOT NULL) OR "
            "(authority_kind <> 'gateway' AND tenant_id IS NULL "
            "AND principal_user_id IS NULL)",
        )
        op.create_foreign_key(
            f"fk_{table_name}_commerce_membership",
            table_name,
            "tenant_memberships",
            ["tenant_id", "principal_user_id"],
            ["tenant_id", "user_id"],
        )
        op.execute(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table_name}_commerce_isolation ON {table_name}
            USING (
                (
                    authority_kind = 'gateway'
                    AND tenant_id = current_setting('app.tenant_id', true)
                    AND principal_user_id =
                        current_setting('app.principal_user_id', true)
                    AND COALESCE(
                        current_setting('app.commerce_authority', true), ''
                    ) = ''
                )
                OR (
                    authority_kind IN ('acp_internal', 'system_internal')
                    AND session_user <> 'icmfyi_payment_worker'
                    AND authority_kind =
                        current_setting('app.commerce_authority', true)
                    AND COALESCE(
                        current_setting('app.tenant_id', true), ''
                    ) = ''
                    AND COALESCE(
                        current_setting('app.principal_user_id', true), ''
                    ) = ''
                )
            )
            WITH CHECK (
                (
                    authority_kind = 'gateway'
                    AND tenant_id = current_setting('app.tenant_id', true)
                    AND principal_user_id =
                        current_setting('app.principal_user_id', true)
                    AND COALESCE(
                        current_setting('app.commerce_authority', true), ''
                    ) = ''
                )
                OR (
                    authority_kind IN ('acp_internal', 'system_internal')
                    AND session_user <> 'icmfyi_payment_worker'
                    AND authority_kind =
                        current_setting('app.commerce_authority', true)
                    AND COALESCE(
                        current_setting('app.tenant_id', true), ''
                    ) = ''
                    AND COALESCE(
                        current_setting('app.principal_user_id', true), ''
                    ) = ''
                )
            )
            """
        )


def _create_postgres_lineage_triggers() -> None:
    op.execute(
        """
        CREATE FUNCTION public.icmfyi_enforce_commerce_lineage()
        RETURNS trigger
        LANGUAGE plpgsql
        SECURITY INVOKER
        SET search_path = pg_catalog, public
        AS $function$
        DECLARE
            parent_authority text;
            parent_tenant text;
            parent_principal text;
            request_document jsonb;
            delivery_document jsonb;
            quote_ids text[];
            line_item_quote_ids text[];
            json_id text;
        BEGIN
            IF TG_OP = 'UPDATE'
               AND ROW(
                   NEW.authority_kind,
                   NEW.tenant_id,
                   NEW.principal_user_id
               ) IS DISTINCT FROM ROW(
                   OLD.authority_kind,
                   OLD.tenant_id,
                   OLD.principal_user_id
               ) THEN
                RAISE EXCEPTION 'commerce ownership is immutable'
                    USING ERRCODE = '23514';
            END IF;

            IF TG_TABLE_NAME = 'channel_packs' THEN
                RETURN NEW;
            END IF;

            IF TG_TABLE_NAME = 'channel_quotes' THEN
                request_document := to_jsonb(NEW.request_json);
                IF jsonb_typeof(request_document) <> 'object' THEN
                    RAISE EXCEPTION 'quote request_json must be an object'
                        USING ERRCODE = '23514';
                END IF;
                IF request_document ? 'pack_id'
                   AND jsonb_typeof(request_document -> 'pack_id')
                       NOT IN ('string', 'null') THEN
                    RAISE EXCEPTION 'quote request_json.pack_id must be a string id'
                        USING ERRCODE = '23514';
                END IF;
                json_id := NULLIF(request_document ->> 'pack_id', '');
                IF json_id IS NOT NULL THEN
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.channel_packs WHERE id = json_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'quote request_json crosses pack ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;
                RETURN NEW;
            END IF;

            IF TG_TABLE_NAME = 'checkout_sessions' THEN
                request_document := to_jsonb(NEW.quote_ids_json);
                delivery_document := to_jsonb(NEW.line_items_json);
                IF jsonb_typeof(request_document) <> 'array'
                   OR jsonb_array_length(request_document) = 0 THEN
                    RAISE EXCEPTION 'checkout quote_ids_json must be a nonempty array'
                        USING ERRCODE = '23514';
                END IF;
                IF EXISTS (
                    SELECT 1 FROM jsonb_array_elements(request_document) AS item(value)
                    WHERE jsonb_typeof(item.value) <> 'string'
                       OR item.value #>> '{}' = ''
                ) THEN
                    RAISE EXCEPTION 'checkout quote_ids_json must contain string ids'
                        USING ERRCODE = '23514';
                END IF;
                IF (
                    SELECT count(*) <> count(DISTINCT item.value #>> '{}')
                    FROM jsonb_array_elements(request_document) AS item(value)
                ) THEN
                    RAISE EXCEPTION 'checkout quote_ids_json contains duplicate ids'
                        USING ERRCODE = '23514';
                END IF;
                IF jsonb_typeof(delivery_document) <> 'array' THEN
                    RAISE EXCEPTION 'checkout line_items_json must be an array'
                        USING ERRCODE = '23514';
                END IF;
                IF EXISTS (
                    SELECT 1 FROM jsonb_array_elements(delivery_document) AS item(value)
                    WHERE jsonb_typeof(item.value) <> 'object'
                       OR jsonb_typeof(item.value -> 'quote_id')
                           IS DISTINCT FROM 'string'
                       OR item.value ->> 'quote_id' = ''
                ) THEN
                    RAISE EXCEPTION 'checkout line items require string quote_id values'
                        USING ERRCODE = '23514';
                END IF;
                SELECT array_agg(item.value #>> '{}' ORDER BY item.ordinality)
                INTO quote_ids
                FROM jsonb_array_elements(request_document)
                    WITH ORDINALITY AS item(value, ordinality);
                SELECT array_agg(item.value ->> 'quote_id' ORDER BY item.ordinality)
                INTO line_item_quote_ids
                FROM jsonb_array_elements(delivery_document)
                    WITH ORDINALITY AS item(value, ordinality);
                IF line_item_quote_ids IS DISTINCT FROM quote_ids THEN
                    RAISE EXCEPTION 'checkout line item quote ids must exactly match quote_ids_json'
                        USING ERRCODE = '23514';
                END IF;
                IF EXISTS (
                    SELECT 1
                    FROM unnest(quote_ids) AS requested_quote(quote_id)
                    LEFT JOIN public.channel_quotes AS parent
                      ON parent.id = requested_quote.quote_id
                    WHERE parent.id IS NULL
                       OR ROW(
                           NEW.authority_kind,
                           NEW.tenant_id,
                           NEW.principal_user_id
                       ) IS DISTINCT FROM ROW(
                           parent.authority_kind,
                           parent.tenant_id,
                           parent.principal_user_id
                       )
                ) THEN
                    RAISE EXCEPTION 'checkout session crosses commerce ownership'
                        USING ERRCODE = '23514';
                END IF;
                RETURN NEW;
            END IF;

            IF TG_TABLE_NAME = 'quote_videos' THEN
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.channel_quotes WHERE id = NEW.quote_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'quote video crosses commerce ownership'
                        USING ERRCODE = '23514';
                END IF;
                RETURN NEW;
            END IF;

            IF TG_TABLE_NAME = 'pack_batches' THEN
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.channel_packs WHERE id = NEW.pack_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'pack batch crosses pack ownership'
                        USING ERRCODE = '23514';
                END IF;
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.channel_quotes WHERE id = NEW.quote_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'pack batch crosses quote ownership'
                        USING ERRCODE = '23514';
                END IF;
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.checkout_sessions WHERE id = NEW.checkout_session_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'pack batch crosses checkout ownership'
                        USING ERRCODE = '23514';
                END IF;
                RETURN NEW;
            END IF;

            IF TG_TABLE_NAME = 'pack_videos' THEN
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.channel_packs WHERE id = NEW.pack_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'pack video crosses pack ownership'
                        USING ERRCODE = '23514';
                END IF;
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.pack_batches WHERE id = NEW.batch_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'pack video crosses batch ownership'
                        USING ERRCODE = '23514';
                END IF;
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.channel_quotes WHERE id = NEW.quote_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'pack video crosses quote ownership'
                        USING ERRCODE = '23514';
                END IF;
                RETURN NEW;
            END IF;

            IF TG_TABLE_NAME = 'channel_orders' THEN
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.channel_quotes WHERE id = NEW.quote_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'order crosses quote ownership'
                        USING ERRCODE = '23514';
                END IF;
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.checkout_sessions WHERE id = NEW.checkout_session_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'order crosses checkout ownership'
                        USING ERRCODE = '23514';
                END IF;
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.channel_packs WHERE id = NEW.pack_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'order crosses pack ownership'
                        USING ERRCODE = '23514';
                END IF;
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.pack_batches WHERE id = NEW.batch_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'order crosses batch ownership'
                        USING ERRCODE = '23514';
                END IF;
                RETURN NEW;
            END IF;

            IF TG_TABLE_NAME = 'payment_receipts' THEN
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.checkout_sessions WHERE id = NEW.checkout_session_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'payment receipt crosses checkout ownership'
                        USING ERRCODE = '23514';
                END IF;
                IF NEW.order_id IS NOT NULL THEN
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.channel_orders WHERE id = NEW.order_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'payment receipt crosses order ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;
                RETURN NEW;
            END IF;

            IF TG_TABLE_NAME = 'acp_job_bridges' THEN
                request_document := to_jsonb(NEW.request_json);
                delivery_document := to_jsonb(NEW.delivery_json);
                IF jsonb_typeof(request_document) <> 'object'
                   OR jsonb_typeof(delivery_document) <> 'object' THEN
                    RAISE EXCEPTION 'ACP bridge request and delivery must be objects'
                        USING ERRCODE = '23514';
                END IF;
                IF request_document ? 'acp_job_id'
                   AND jsonb_typeof(request_document -> 'acp_job_id')
                       NOT IN ('string', 'null') THEN
                    RAISE EXCEPTION 'ACP request acp_job_id must be a string id'
                        USING ERRCODE = '23514';
                END IF;
                IF delivery_document ? 'acp_job_id'
                   AND jsonb_typeof(delivery_document -> 'acp_job_id')
                       NOT IN ('string', 'null') THEN
                    RAISE EXCEPTION 'ACP delivery acp_job_id must be a string id'
                        USING ERRCODE = '23514';
                END IF;
                IF NULLIF(request_document ->> 'acp_job_id', '') IS NOT NULL
                   AND request_document ->> 'acp_job_id' <> NEW.acp_job_id THEN
                    RAISE EXCEPTION 'ACP request acp_job_id disagrees with bridge id'
                        USING ERRCODE = '23514';
                END IF;
                IF NULLIF(delivery_document ->> 'acp_job_id', '') IS NOT NULL
                   AND delivery_document ->> 'acp_job_id' <> NEW.acp_job_id THEN
                    RAISE EXCEPTION 'ACP delivery acp_job_id disagrees with bridge id'
                        USING ERRCODE = '23514';
                END IF;
                IF request_document ? 'pack_id'
                   AND jsonb_typeof(request_document -> 'pack_id')
                       NOT IN ('string', 'null') THEN
                    RAISE EXCEPTION 'ACP request pack_id must be a string id'
                        USING ERRCODE = '23514';
                END IF;
                IF EXISTS (
                    SELECT 1
                    FROM (VALUES
                        ('quote_id'), ('order_id'), ('pack_id'), ('batch_id')
                    ) AS required_type(key_name)
                    WHERE delivery_document ? required_type.key_name
                      AND jsonb_typeof(delivery_document -> required_type.key_name)
                          NOT IN ('string', 'null')
                ) THEN
                    RAISE EXCEPTION 'ACP delivery commerce ids must be string ids'
                        USING ERRCODE = '23514';
                END IF;
                IF NEW.quote_id IS NULL
                   AND NEW.checkout_session_id IS NULL
                   AND NEW.order_id IS NULL
                   AND NEW.pack_id IS NULL
                   AND NULLIF(request_document ->> 'pack_id', '') IS NULL
                   AND NULLIF(delivery_document ->> 'quote_id', '') IS NULL
                   AND NULLIF(delivery_document ->> 'order_id', '') IS NULL
                   AND NULLIF(delivery_document ->> 'pack_id', '') IS NULL
                   AND NULLIF(delivery_document ->> 'batch_id', '') IS NULL THEN
                    RAISE EXCEPTION 'ACP bridge is detached from commerce lifecycle'
                        USING ERRCODE = '23514';
                END IF;
                IF NEW.quote_id IS NOT NULL THEN
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.channel_quotes WHERE id = NEW.quote_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'ACP bridge crosses quote ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;
                IF NEW.checkout_session_id IS NOT NULL THEN
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.checkout_sessions WHERE id = NEW.checkout_session_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'ACP bridge crosses checkout ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;
                IF NEW.order_id IS NOT NULL THEN
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.channel_orders WHERE id = NEW.order_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'ACP bridge crosses order ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;
                IF NEW.pack_id IS NOT NULL THEN
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.channel_packs WHERE id = NEW.pack_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'ACP bridge crosses pack ownership'
                        USING ERRCODE = '23514';
                    END IF;
                END IF;

                json_id := NULLIF(request_document ->> 'pack_id', '');
                IF json_id IS NOT NULL THEN
                    IF NEW.pack_id IS NOT NULL AND NEW.pack_id <> json_id THEN
                        RAISE EXCEPTION 'ACP request pack_id disagrees with bridge pack_id'
                            USING ERRCODE = '23514';
                    END IF;
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.channel_packs WHERE id = json_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'ACP request crosses pack ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;

                json_id := NULLIF(delivery_document ->> 'quote_id', '');
                IF json_id IS NOT NULL THEN
                    IF NEW.quote_id IS NOT NULL AND NEW.quote_id <> json_id THEN
                        RAISE EXCEPTION 'ACP delivery quote_id disagrees with bridge quote_id'
                            USING ERRCODE = '23514';
                    END IF;
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.channel_quotes WHERE id = json_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'ACP delivery crosses quote ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;

                json_id := NULLIF(delivery_document ->> 'order_id', '');
                IF json_id IS NOT NULL THEN
                    IF NEW.order_id IS NOT NULL AND NEW.order_id <> json_id THEN
                        RAISE EXCEPTION 'ACP delivery order_id disagrees with bridge order_id'
                            USING ERRCODE = '23514';
                    END IF;
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.channel_orders WHERE id = json_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'ACP delivery crosses order ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;

                json_id := NULLIF(delivery_document ->> 'pack_id', '');
                IF json_id IS NOT NULL THEN
                    IF NEW.pack_id IS NOT NULL AND NEW.pack_id <> json_id THEN
                        RAISE EXCEPTION 'ACP delivery pack_id disagrees with bridge pack_id'
                            USING ERRCODE = '23514';
                    END IF;
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.channel_packs WHERE id = json_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'ACP delivery crosses pack ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;

                json_id := NULLIF(delivery_document ->> 'batch_id', '');
                IF json_id IS NOT NULL THEN
                    SELECT authority_kind, tenant_id, principal_user_id
                    INTO parent_authority, parent_tenant, parent_principal
                    FROM public.pack_batches WHERE id = json_id;
                    IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                        IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                        RAISE EXCEPTION 'ACP delivery crosses batch ownership'
                            USING ERRCODE = '23514';
                    END IF;
                END IF;
                RETURN NEW;
            END IF;

            IF TG_TABLE_NAME = 'entitlements' THEN
                SELECT authority_kind, tenant_id, principal_user_id
                INTO parent_authority, parent_tenant, parent_principal
                FROM public.channel_packs WHERE id = NEW.pack_id;
                IF NOT FOUND OR ROW(NEW.authority_kind, NEW.tenant_id, NEW.principal_user_id)
                    IS DISTINCT FROM ROW(parent_authority, parent_tenant, parent_principal) THEN
                    RAISE EXCEPTION 'entitlement crosses pack ownership'
                        USING ERRCODE = '23514';
                END IF;
                RETURN NEW;
            END IF;

            RAISE EXCEPTION 'unsupported commerce lineage trigger table'
                USING ERRCODE = '23514';
        END
        $function$
        """
    )
    for table_name in (
        "channel_quotes",
        "quote_videos",
        "checkout_sessions",
        "channel_packs",
        "pack_batches",
        "pack_videos",
        "channel_orders",
        "payment_receipts",
        "acp_job_bridges",
        "entitlements",
    ):
        op.execute(
            f"CREATE TRIGGER {table_name}_commerce_lineage "
            f"BEFORE INSERT OR UPDATE ON {table_name} FOR EACH ROW "
            "EXECUTE FUNCTION public.icmfyi_enforce_commerce_lineage()"
        )


def upgrade() -> None:
    connection = op.get_bind()
    is_sqlite = connection.dialect.name == "sqlite"
    if is_sqlite:
        # Alembic treats SQLite DDL as non-transactional. Validate all legacy
        # edges before the first schema mutation and tolerate columns left by an
        # interrupted prior attempt so a repaired database can retry safely.
        _load_commerce_graph(connection, COMMERCE_TABLES, include_ownership=False)
        inspector = sa.inspect(connection)
        existing_tables = set(inspector.get_table_names())
        existing_columns = {
            table_name: {
                str(column["name"])
                for column in inspector.get_columns(table_name)
            }
            for table_name in COMMERCE_TABLES
        }
    else:
        existing_tables = set()
        existing_columns = {table_name: set() for table_name in COMMERCE_TABLES}

    _replace_source_video_archive_state_constraint(_SOURCE_VIDEO_ARCHIVE_STATES_V5)

    for table_name in COMMERCE_TABLES:
        for column in (
            sa.Column("authority_kind", sa.String(length=32), nullable=True),
            sa.Column("tenant_id", sa.String(length=68), nullable=True),
            sa.Column("principal_user_id", sa.String(length=68), nullable=True),
        ):
            if column.name not in existing_columns[table_name]:
                op.add_column(table_name, column)
        op.execute(
            f"UPDATE {table_name} SET authority_kind = 'legacy_internal' "
            "WHERE authority_kind IS NULL"
        )

    if "commerce_json" not in existing_columns["channel_quotes"]:
        op.add_column(
            "channel_quotes", sa.Column("commerce_json", sa.JSON(), nullable=True)
        )
    quotes = sa.table("channel_quotes", sa.column("commerce_json", sa.JSON()))
    connection.execute(
        sa.update(quotes)
        .where(quotes.c.commerce_json.is_(None))
        .values(commerce_json={})
    )

    _reconcile_commerce_lineage(connection, COMMERCE_TABLES)
    _create_scheduler_quote_projection(
        connection,
        existing="scheduler_quote_video_projection" in existing_tables,
    )

    op.drop_index(
        "ix_checkout_sessions_idempotency_key", table_name="checkout_sessions"
    )
    op.create_index(
        "ix_checkout_sessions_idempotency_key",
        "checkout_sessions",
        ["idempotency_key"],
    )
    op.create_index(
        "uq_checkout_sessions_gateway_idempotency",
        "checkout_sessions",
        ["tenant_id", "principal_user_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("authority_kind = 'gateway'"),
        sqlite_where=sa.text("authority_kind = 'gateway'"),
    )
    op.create_index(
        "uq_checkout_sessions_internal_idempotency",
        "checkout_sessions",
        ["authority_kind", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text(
            "authority_kind IN ('acp_internal', 'system_internal')"
        ),
        sqlite_where=sa.text("authority_kind IN ('acp_internal', 'system_internal')"),
    )

    for table_name in COMMERCE_TABLES:
        for column_name in ("authority_kind", "tenant_id", "principal_user_id"):
            op.create_index(f"ix_{table_name}_{column_name}", table_name, [column_name])

    if connection.dialect.name == "postgresql":
        op.alter_column("channel_quotes", "commerce_json", nullable=False)
        _create_postgres_constraints_and_policies()
        _create_postgres_lineage_triggers()
        _create_postgres_scheduler_projection_triggers()


def downgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        # FORCE RLS also applies to a NOBYPASSRLS table owner.  Lift it inside
        # the migration transaction so the destructive downgrade check sees
        # every commerce row; a refused downgrade rolls this DDL back and
        # leaves the policies forced.
        for table_name in COMMERCE_TABLES:
            op.execute(f"ALTER TABLE {table_name} NO FORCE ROW LEVEL SECURITY")
    _assert_downgrade_reconstructible(connection)
    blocked_count = int(
        connection.execute(
            sa.text(
                "SELECT count(*) FROM source_videos "
                "WHERE archive_state = 'blocked_public_age_gate'"
            )
        ).scalar_one()
    )
    if blocked_count:
        raise RuntimeError(
            "downgrade would invalidate "
            f"{blocked_count} blocked_public_age_gate source video rows"
        )

    if connection.dialect.name == "postgresql":
        op.execute(
            "DROP TRIGGER IF EXISTS quote_videos_scheduler_projection "
            "ON quote_videos"
        )
        op.execute(
            "DROP TRIGGER IF EXISTS scheduler_quote_video_projection_guard "
            "ON scheduler_quote_video_projection"
        )
        op.execute(
            "DROP FUNCTION IF EXISTS "
            "public.icmfyi_guard_scheduler_quote_video_projection()"
        )
        op.execute(
            "DROP FUNCTION IF EXISTS "
            "public.icmfyi_sync_scheduler_quote_video_projection()"
        )
        for table_name in reversed(
            (
                "channel_quotes",
                "quote_videos",
                "checkout_sessions",
                "channel_packs",
                "pack_batches",
                "pack_videos",
                "channel_orders",
                "payment_receipts",
                "acp_job_bridges",
                "entitlements",
            )
        ):
            op.execute(
                f"DROP TRIGGER IF EXISTS {table_name}_commerce_lineage ON {table_name}"
            )
        op.execute("DROP FUNCTION IF EXISTS public.icmfyi_enforce_commerce_lineage()")
        for table_name in reversed(COMMERCE_TABLES):
            op.execute(
                f"DROP POLICY IF EXISTS {table_name}_commerce_isolation ON {table_name}"
            )
            op.execute(f"ALTER TABLE {table_name} DISABLE ROW LEVEL SECURITY")
            op.drop_constraint(
                f"fk_{table_name}_commerce_membership",
                table_name,
                type_="foreignkey",
            )
            op.drop_constraint(
                f"ck_{table_name}_commerce_owner_shape",
                table_name,
                type_="check",
            )
            op.drop_constraint(
                f"ck_{table_name}_commerce_authority",
                table_name,
                type_="check",
            )

    op.drop_index(
        "ix_scheduler_quote_video_projection_status",
        table_name="scheduler_quote_video_projection",
    )
    op.drop_index(
        "ix_scheduler_quote_video_projection_video_id",
        table_name="scheduler_quote_video_projection",
    )
    op.drop_table("scheduler_quote_video_projection")

    _replace_source_video_archive_state_constraint(_SOURCE_VIDEO_ARCHIVE_STATES_V4)

    for table_name in reversed(COMMERCE_TABLES):
        for column_name in reversed(
            ("authority_kind", "tenant_id", "principal_user_id")
        ):
            op.drop_index(f"ix_{table_name}_{column_name}", table_name=table_name)

    op.drop_index(
        "uq_checkout_sessions_internal_idempotency",
        table_name="checkout_sessions",
    )
    op.drop_index(
        "uq_checkout_sessions_gateway_idempotency",
        table_name="checkout_sessions",
    )
    op.drop_index(
        "ix_checkout_sessions_idempotency_key", table_name="checkout_sessions"
    )
    op.create_index(
        "ix_checkout_sessions_idempotency_key",
        "checkout_sessions",
        ["idempotency_key"],
        unique=True,
    )

    op.drop_column("channel_quotes", "commerce_json")
    for table_name in reversed(COMMERCE_TABLES):
        op.drop_column(table_name, "principal_user_id")
        op.drop_column(table_name, "tenant_id")
        op.drop_column(table_name, "authority_kind")
