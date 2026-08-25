\set ON_ERROR_STOP on

-- Cross-tenant payment dequeue is exposed only through these three static
-- functions.  The dedicated worker receives EXECUTE, never direct privileges
-- on payment_work_intents, payment_settlements, or payment_work_outbox.
-- Apply as the database owner after both the ingestion migrations and the
-- x402 payment schema exist.  Never transfer function ownership to a runtime
-- login and always REVOKE PUBLIC before granting the worker role.

BEGIN;

CREATE OR REPLACE FUNCTION public.icmfyi_claim_settled_paid_work()
RETURNS TABLE (
    outbox_id uuid,
    intent_id uuid,
    tenant_id text,
    principal_id text,
    topic text,
    idempotency_key text,
    request_hash text,
    tool_name text,
    commerce_quote_id text,
    commerce_quote_hash text,
    asset text,
    amount_atomic numeric,
    payload jsonb,
    settlement_network text,
    settlement_transaction text,
    settlement_recorded_at timestamptz
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    claimed record;
BEGIN
    IF COALESCE(current_setting('icmfyi.payment_work_claim_id', true), '') <> '' THEN
        RAISE EXCEPTION 'one paid-work claim is already active in this transaction';
    END IF;

    SELECT
        outbox.id AS claimed_outbox_id,
        intent.id AS claimed_intent_id,
        intent.tenant_id AS claimed_tenant_id,
        intent.principal_id AS claimed_principal_id,
        outbox.topic AS claimed_topic,
        intent.idempotency_key AS claimed_idempotency_key,
        intent.request_hash AS claimed_request_hash,
        intent.tool_name AS claimed_tool_name,
        intent.commerce_quote_id AS claimed_commerce_quote_id,
        intent.commerce_quote_hash AS claimed_commerce_quote_hash,
        intent.asset AS claimed_asset,
        intent.amount_atomic AS claimed_amount_atomic,
        outbox.payload AS claimed_payload,
        settlement.network AS claimed_settlement_network,
        settlement.transaction AS claimed_settlement_transaction,
        settlement.recorded_at AS claimed_settlement_recorded_at
    INTO claimed
    FROM public.payment_work_outbox AS outbox
    JOIN public.payment_work_intents AS intent
      ON intent.id = outbox.intent_id
     AND intent.tenant_id = outbox.tenant_id
     AND intent.idempotency_key = outbox.idempotency_key
     AND intent.request_hash = outbox.request_hash
    JOIN public.payment_settlements AS settlement
      ON settlement.intent_id = intent.id
     AND settlement.tenant_id = intent.tenant_id
     AND settlement.request_hash = intent.request_hash
    WHERE outbox.published_at IS NULL
      AND outbox.dead_lettered_at IS NULL
      AND outbox.next_delivery_attempt_at <= pg_catalog.clock_timestamp()
      AND outbox.topic = 'icmfyi.work.requested.v1'
      AND intent.status = 'settled'
      AND intent.tool_name = 'icmfyi.ingest.youtube'
      AND outbox.payload ->> 'schema' = 'icmfyi.paid-work-request.v1'
      AND outbox.payload ->> 'tenantId' = intent.tenant_id
      AND outbox.payload ->> 'principalId' = intent.principal_id
      AND outbox.payload ->> 'toolName' = intent.tool_name
      AND outbox.payload ->> 'idempotencyKey' = intent.idempotency_key
      AND outbox.payload ->> 'requestHash' = intent.request_hash
      AND outbox.payload -> 'commerce' ->> 'provider' = 'icmfyi-acp'
      AND outbox.payload -> 'commerce' ->> 'quoteId' = intent.commerce_quote_id
      AND outbox.payload -> 'commerce' ->> 'quoteHash' = intent.commerce_quote_hash
      AND outbox.payload -> 'work' ->> 'schema' = 'icmfyi.channel-pack-work.v1'
      AND outbox.payload -> 'work' ->> 'operation' =
          'create_settled_channel_pack_order'
      AND outbox.payload -> 'work' ->> 'quoteId' = intent.commerce_quote_id
    ORDER BY
        outbox.next_delivery_attempt_at ASC,
        outbox.delivery_attempt_count ASC,
        outbox.created_at ASC,
        outbox.id ASC
    FOR UPDATE OF outbox SKIP LOCKED
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN;
    END IF;

    -- Serialize all outbox rows that target the same authoritative quote
    -- without granting the worker UPDATE solely for SELECT .. FOR UPDATE.
    -- A busy quote is skipped for this transaction and remains unpublished.
    IF NOT pg_catalog.pg_try_advisory_xact_lock(
        pg_catalog.hashtextextended(claimed.claimed_commerce_quote_id, 0)
    ) THEN
        RETURN;
    END IF;

    PERFORM pg_catalog.set_config(
        'icmfyi.payment_work_claim_id', claimed.claimed_outbox_id::text, true
    );
    PERFORM pg_catalog.set_config(
        'icmfyi.payment_work_claim_intent_id', claimed.claimed_intent_id::text, true
    );
    PERFORM pg_catalog.set_config(
        'icmfyi.tenant_id', claimed.claimed_tenant_id, true
    );

    RETURN QUERY SELECT
        claimed.claimed_outbox_id,
        claimed.claimed_intent_id,
        claimed.claimed_tenant_id,
        claimed.claimed_principal_id,
        claimed.claimed_topic,
        claimed.claimed_idempotency_key,
        claimed.claimed_request_hash,
        claimed.claimed_tool_name,
        claimed.claimed_commerce_quote_id,
        claimed.claimed_commerce_quote_hash,
        claimed.claimed_asset,
        claimed.claimed_amount_atomic,
        claimed.claimed_payload,
        claimed.claimed_settlement_network,
        claimed.claimed_settlement_transaction,
        claimed.claimed_settlement_recorded_at;
END
$function$;

CREATE OR REPLACE FUNCTION public.icmfyi_ack_settled_paid_work(
    requested_outbox_id uuid,
    requested_intent_id uuid,
    created_order_id text
)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    acknowledged boolean := false;
BEGIN
    IF current_setting('icmfyi.payment_work_claim_id', true)
           IS DISTINCT FROM requested_outbox_id::text
       OR current_setting('icmfyi.payment_work_claim_intent_id', true)
           IS DISTINCT FROM requested_intent_id::text THEN
        RAISE EXCEPTION 'paid-work acknowledgement does not match this transaction claim';
    END IF;

    UPDATE public.payment_work_outbox AS outbox
    SET published_at = pg_catalog.clock_timestamp(),
        delivery_attempt_count = outbox.delivery_attempt_count + 1,
        last_delivery_error_code = NULL,
        last_delivery_error_detail = NULL
    FROM public.payment_work_intents AS intent,
         public.payment_settlements AS settlement
    WHERE outbox.id = requested_outbox_id
      AND outbox.intent_id = requested_intent_id
      AND outbox.published_at IS NULL
      AND outbox.dead_lettered_at IS NULL
      AND outbox.topic = 'icmfyi.work.requested.v1'
      AND intent.id = outbox.intent_id
      AND intent.tenant_id = outbox.tenant_id
      AND intent.idempotency_key = outbox.idempotency_key
      AND intent.request_hash = outbox.request_hash
      AND intent.status = 'settled'
      AND intent.tool_name = 'icmfyi.ingest.youtube'
      AND settlement.intent_id = intent.id
      AND settlement.tenant_id = intent.tenant_id
      AND settlement.request_hash = intent.request_hash
      AND EXISTS (
          SELECT 1
          FROM public.channel_orders AS channel_order
          JOIN public.payment_receipts AS receipt
            ON receipt.order_id = channel_order.id
           AND receipt.authority_kind = channel_order.authority_kind
           AND receipt.tenant_id = channel_order.tenant_id
           AND receipt.principal_user_id = channel_order.principal_user_id
           AND receipt.checkout_session_id = channel_order.checkout_session_id
          JOIN public.pack_batches AS batch
            ON batch.id = channel_order.batch_id
           AND batch.pack_id = channel_order.pack_id
           AND batch.quote_id = channel_order.quote_id
           AND batch.checkout_session_id = channel_order.checkout_session_id
           AND batch.authority_kind = channel_order.authority_kind
           AND batch.tenant_id = channel_order.tenant_id
           AND batch.principal_user_id = channel_order.principal_user_id
          JOIN public.channel_quotes AS quote
            ON quote.id = channel_order.quote_id
           AND quote.authority_kind = channel_order.authority_kind
           AND quote.tenant_id = channel_order.tenant_id
           AND quote.principal_user_id = channel_order.principal_user_id
          WHERE channel_order.id = created_order_id
            AND channel_order.authority_kind = 'gateway'
            AND channel_order.tenant_id = intent.tenant_id
            AND channel_order.principal_user_id = intent.principal_id
            AND channel_order.quote_id = intent.commerce_quote_id
            AND quote.commerce_json ->> 'paymentIdempotencyKey' =
                intent.idempotency_key
            AND channel_order.payment_provider = 'x402'
            AND channel_order.payment_status = 'settled_x402'
            AND receipt.provider = 'x402'
            AND receipt.status = 'settled'
            AND receipt.receipt_json ->> 'schema' =
                'icmfyi.x402-settlement-receipt.v1'
            AND receipt.receipt_json ->> 'paymentIntentId' = intent.id::text
            AND receipt.receipt_json ->> 'outboxId' = outbox.id::text
            AND receipt.receipt_json ->> 'idempotencyKey' = intent.idempotency_key
            AND receipt.receipt_json ->> 'requestHash' = intent.request_hash
            AND receipt.receipt_json ->> 'quoteHash' = intent.commerce_quote_hash
            AND receipt.receipt_json ->> 'network' = settlement.network
            AND receipt.receipt_json ->> 'transaction' = settlement.transaction
            AND batch.batch_index = quote.current_batch_index
            AND batch.billable_video_count = quote.current_batch_video_count
            AND quote.current_batch_video_count > 0
            AND (
                SELECT pg_catalog.count(*)
                FROM public.quote_videos AS quote_video
                WHERE quote_video.quote_id = quote.id
                  AND quote_video.included
                  AND quote_video.batch_index = quote.current_batch_index
                  AND quote_video.authority_kind = quote.authority_kind
                  AND quote_video.tenant_id = quote.tenant_id
                  AND quote_video.principal_user_id = quote.principal_user_id
            ) = quote.current_batch_video_count
            AND (
                SELECT pg_catalog.count(*)
                FROM public.pack_videos AS pack_video
                WHERE pack_video.batch_id = channel_order.batch_id
                  AND pack_video.pack_id = channel_order.pack_id
                  AND pack_video.quote_id = channel_order.quote_id
                  AND pack_video.authority_kind = channel_order.authority_kind
                  AND pack_video.tenant_id = channel_order.tenant_id
                  AND pack_video.principal_user_id = channel_order.principal_user_id
            ) = quote.current_batch_video_count
            AND NOT EXISTS (
                SELECT 1
                FROM public.quote_videos AS quote_video
                WHERE quote_video.quote_id = quote.id
                  AND quote_video.included
                  AND quote_video.batch_index = quote.current_batch_index
                  AND quote_video.authority_kind = quote.authority_kind
                  AND quote_video.tenant_id = quote.tenant_id
                  AND quote_video.principal_user_id = quote.principal_user_id
                GROUP BY quote_video.position
                HAVING pg_catalog.count(*) <> 1
            )
            AND NOT EXISTS (
                SELECT 1
                FROM public.pack_videos AS pack_video
                WHERE pack_video.batch_id = channel_order.batch_id
                  AND pack_video.pack_id = channel_order.pack_id
                  AND pack_video.quote_id = channel_order.quote_id
                  AND pack_video.authority_kind = channel_order.authority_kind
                  AND pack_video.tenant_id = channel_order.tenant_id
                  AND pack_video.principal_user_id = channel_order.principal_user_id
                GROUP BY pack_video.position
                HAVING pg_catalog.count(*) <> 1
            )
            AND NOT EXISTS (
                SELECT 1
                FROM public.quote_videos AS quote_video
                WHERE quote_video.quote_id = quote.id
                  AND quote_video.included
                  AND quote_video.batch_index = quote.current_batch_index
                  AND quote_video.authority_kind = quote.authority_kind
                  AND quote_video.tenant_id = quote.tenant_id
                  AND quote_video.principal_user_id = quote.principal_user_id
                  AND NOT EXISTS (
                      SELECT 1
                      FROM public.pack_videos AS pack_video
                      WHERE pack_video.batch_id = channel_order.batch_id
                        AND pack_video.pack_id = channel_order.pack_id
                        AND pack_video.quote_id = channel_order.quote_id
                        AND pack_video.authority_kind = channel_order.authority_kind
                        AND pack_video.tenant_id = channel_order.tenant_id
                        AND pack_video.principal_user_id = channel_order.principal_user_id
                        AND pack_video.position = quote_video.position
                        AND pack_video.video_id = quote_video.video_id
                  )
            )
            AND NOT EXISTS (
                SELECT 1
                FROM public.pack_videos AS pack_video
                WHERE pack_video.batch_id = channel_order.batch_id
                  AND pack_video.pack_id = channel_order.pack_id
                  AND pack_video.quote_id = channel_order.quote_id
                  AND pack_video.authority_kind = channel_order.authority_kind
                  AND pack_video.tenant_id = channel_order.tenant_id
                  AND pack_video.principal_user_id = channel_order.principal_user_id
                  AND NOT EXISTS (
                      SELECT 1
                      FROM public.ingestion_requests AS request
                      JOIN public.ingestion_jobs AS job ON job.id = request.job_id
                      WHERE request.tenant_id = intent.tenant_id
                        AND request.requested_by_user_id = intent.principal_id
                        AND request.status IN ('accepted', 'ready')
                        AND job.job_kind = 'public_item_ingestion'
                        AND job.source_kind = 'youtube'
                        AND pg_catalog.right(
                            job.source_key,
                            pg_catalog.length(pack_video.video_id) + 1
                        ) = ':' || pack_video.video_id
                        AND request.request_json ->> 'schema' =
                            'icmfyi.public-ingestion-request.v1'
                        AND request.request_json -> 'item' ->> 'platform' = 'youtube'
                        AND request.request_json -> 'item' ->> 'external_id' =
                            pack_video.video_id
                        AND request.request_json -> 'paidWork' ->> 'schema' =
                            'icmfyi.paid-public-ingestion.v1'
                        AND request.request_json -> 'paidWork' ->> 'intentId' =
                            intent.id::text
                        AND request.request_json -> 'paidWork' ->> 'outboxId' =
                            outbox.id::text
                        AND request.request_json -> 'paidWork' ->> 'orderId' =
                            channel_order.id
                        AND request.request_json -> 'paidWork' ->> 'packId' =
                            pack_video.pack_id
                        AND request.request_json -> 'paidWork' ->> 'batchId' =
                            pack_video.batch_id
                        AND request.request_json -> 'paidWork' ->> 'quoteId' =
                            pack_video.quote_id
                        AND request.request_json -> 'paidWork' ->> 'videoId' =
                            pack_video.video_id
                        AND request.request_json -> 'paidWork' ->> 'position' =
                            pack_video.position::text
                  )
            )
      )
    RETURNING true INTO acknowledged;

    IF NOT COALESCE(acknowledged, false) THEN
        RAISE EXCEPTION 'paid-work acknowledgement lacks exact settled order readback';
    END IF;

    PERFORM pg_catalog.set_config('icmfyi.payment_work_claim_id', '', true);
    PERFORM pg_catalog.set_config('icmfyi.payment_work_claim_intent_id', '', true);
    RETURN true;
END
$function$;

CREATE OR REPLACE FUNCTION public.icmfyi_fail_settled_paid_work(
    requested_outbox_id uuid,
    requested_intent_id uuid,
    error_code text,
    error_detail text,
    retryable boolean,
    maximum_attempts integer,
    retry_delay_seconds integer
)
RETURNS boolean
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    recorded boolean := false;
BEGIN
    IF current_setting('icmfyi.payment_work_claim_id', true)
           IS DISTINCT FROM requested_outbox_id::text
       OR current_setting('icmfyi.payment_work_claim_intent_id', true)
           IS DISTINCT FROM requested_intent_id::text THEN
        RAISE EXCEPTION 'paid-work failure does not match this transaction claim';
    END IF;
    IF error_code IS NULL OR error_code !~ '^[a-z0-9_]{1,64}$'
       OR maximum_attempts < 1 OR maximum_attempts > 100
       OR retry_delay_seconds < 0 OR retry_delay_seconds > 86400 THEN
        RAISE EXCEPTION 'paid-work failure fields are invalid';
    END IF;

    UPDATE public.payment_work_outbox AS outbox
    SET delivery_attempt_count = outbox.delivery_attempt_count + 1,
        next_delivery_attempt_at = CASE
            WHEN retryable
             AND outbox.delivery_attempt_count + 1 < maximum_attempts
            THEN pg_catalog.clock_timestamp()
                 + pg_catalog.make_interval(secs => retry_delay_seconds)
            ELSE outbox.next_delivery_attempt_at
        END,
        last_delivery_error_code = error_code,
        last_delivery_error_detail = pg_catalog.left(COALESCE(error_detail, ''), 8000),
        dead_lettered_at = CASE
            WHEN retryable
             AND outbox.delivery_attempt_count + 1 < maximum_attempts
            THEN NULL
            ELSE pg_catalog.clock_timestamp()
        END
    WHERE outbox.id = requested_outbox_id
      AND outbox.intent_id = requested_intent_id
      AND outbox.published_at IS NULL
      AND outbox.dead_lettered_at IS NULL
    RETURNING true INTO recorded;

    IF NOT COALESCE(recorded, false) THEN
        RAISE EXCEPTION 'paid-work failure lacks an active unpublished row';
    END IF;

    PERFORM pg_catalog.set_config('icmfyi.payment_work_claim_id', '', true);
    PERFORM pg_catalog.set_config('icmfyi.payment_work_claim_intent_id', '', true);
    RETURN true;
END
$function$;

REVOKE ALL ON FUNCTION public.icmfyi_claim_settled_paid_work() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.icmfyi_ack_settled_paid_work(uuid, uuid, text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.icmfyi_fail_settled_paid_work(
    uuid, uuid, text, text, boolean, integer, integer
) FROM PUBLIC;

COMMENT ON FUNCTION public.icmfyi_claim_settled_paid_work() IS
    'Claims one fully joined settled x402 outbox row under a transaction-held lock.';
COMMENT ON FUNCTION public.icmfyi_ack_settled_paid_work(uuid, uuid, text) IS
    'Acknowledges only an exact tenant-owned order whose billed videos all have durable public-ingestion requests.';
COMMENT ON FUNCTION public.icmfyi_fail_settled_paid_work(
    uuid, uuid, text, text, boolean, integer, integer
) IS
    'Commits bounded retry or dead-letter state for only the active transaction claim.';

COMMIT;
