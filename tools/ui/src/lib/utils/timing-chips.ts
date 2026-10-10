/**
 * W2-quick observability: pure formulas behind the timing chips, the tool-call chips, the turn
 * summary line and the context-bar speed line (spec 2026-10-10-webui-observability-spec.md).
 * Everything is derived from the stored `timings` (server) + the client fields; old messages
 * render with whatever they have.
 */

import { MessageRole } from '$lib/enums';
import type {
	ChatMessageAgenticLlmTotals,
	ChatMessageTimings,
	ChatMessageToolCallTiming,
	DatabaseMessage
} from '$lib/types';

export interface TimingChips {
	/** request wall (client) or the flow's wall */
	totalMs?: number;
	ttftMs?: number;
	prefill?: { fresh: number; cached: number; ms: number; tps?: number; cacheShare?: number };
	decode?: { n: number; ms: number; tps?: number };
	/** MTP acceptance 0..1 */
	mtp?: number;
}

const tps = (n: number, ms: number): number | undefined =>
	n > 0 && ms > 0 ? (n / ms) * 1000 : undefined;

/** The LLM totals a message's chips describe: the flow totals of an agentic root, else the request. */
export function llmTotals(t: ChatMessageTimings | undefined): ChatMessageAgenticLlmTotals | null {
	if (!t) return null;

	if (t.agentic?.llm) return t.agentic.llm;

	if (
		t.prompt_n === undefined &&
		t.predicted_n === undefined &&
		t.prompt_ms === undefined &&
		t.predicted_ms === undefined
	) {
		return null;
	}

	return {
		cache_n: t.cache_n,
		draft_n: t.draft_n,
		draft_n_accepted: t.draft_n_accepted,
		predicted_ms: t.predicted_ms ?? 0,
		predicted_n: t.predicted_n ?? 0,
		prompt_ms: t.prompt_ms ?? 0,
		prompt_n: t.prompt_n ?? 0
	};
}

/** total, TTFT, prefill (fresh + cached, t/s of the fresh tokens, cache share), decode, MTP %. */
export function timingChips(t: ChatMessageTimings | undefined): TimingChips | null {
	const llm = llmTotals(t);

	if (!t || !llm) return null;

	const fresh = llm.prompt_n ?? 0;
	const cached = llm.cache_n ?? 0;
	const out: TimingChips = {};
	const flowWall =
		t.agentic?.t_start !== undefined && t.agentic?.t_end !== undefined
			? t.agentic.t_end - t.agentic.t_start
			: undefined;

	out.totalMs = flowWall ?? t.client?.wall_ms;
	out.ttftMs = t.client?.ttft_ms;

	if (fresh + cached > 0) {
		out.prefill = {
			cached,
			cacheShare: cached / (fresh + cached),
			fresh,
			ms: llm.prompt_ms ?? 0,
			tps: tps(fresh, llm.prompt_ms ?? 0)
		};
	}

	if ((llm.predicted_n ?? 0) > 0) {
		out.decode = {
			ms: llm.predicted_ms ?? 0,
			n: llm.predicted_n,
			tps: tps(llm.predicted_n, llm.predicted_ms ?? 0)
		};
	}

	if ((llm.draft_n ?? 0) > 0) out.mtp = (llm.draft_n_accepted ?? 0) / (llm.draft_n as number);

	return out;
}

export interface TurnSummary {
	wallMs: number;
	modelMs: number;
	prefillMs: number;
	decodeMs: number;
	toolsMs: number;
	otherMs: number;
	llmCalls: number;
	toolCalls: number;
}

/**
 * Turn wall = model (Σ prompt_ms + predicted_ms) + tools (Σ durations) + other. The wall is the
 * flow's own start -> end (agentic), the request wall (plain), else the span of the message
 * timestamps given (user message -> last message). Null without a wall.
 */
export function turnSummary(
	t: ChatMessageTimings | undefined,
	fallback?: { startMs?: number; endMs?: number }
): TurnSummary | null {
	const llm = llmTotals(t);

	if (!t || !llm) return null;

	const prefillMs = llm.prompt_ms ?? 0;
	const decodeMs = llm.predicted_ms ?? 0;
	const toolsMs = t.agentic?.toolsMs ?? 0;
	const flowWall =
		t.agentic?.t_start !== undefined && t.agentic?.t_end !== undefined
			? t.agentic.t_end - t.agentic.t_start
			: undefined;
	const tsWall =
		fallback?.startMs !== undefined && fallback?.endMs !== undefined
			? fallback.endMs - fallback.startMs
			: undefined;
	const wallMs = flowWall ?? t.client?.wall_ms ?? tsWall;

	if (wallMs === undefined || wallMs <= 0) return null;

	const modelMs = prefillMs + decodeMs;

	return {
		decodeMs,
		llmCalls: t.agentic?.turns ?? 1,
		modelMs,
		otherMs: Math.max(0, wallMs - modelMs - toolsMs),
		prefillMs,
		toolCalls: t.agentic?.toolCallsCount ?? 0,
		toolsMs,
		wallMs
	};
}

/** Tool-call records of a flow, by tool-call id (the root's final list + any per-message snapshot). */
export function toolRecordIndex(
	messages: readonly (DatabaseMessage | undefined)[]
): Map<string, ChatMessageToolCallTiming> {
	const out = new Map<string, ChatMessageToolCallTiming>();

	for (const m of messages) {
		for (const r of m?.timings?.agentic?.toolCalls ?? []) {
			if (r.toolCallId) out.set(r.toolCallId, r);
		}
	}

	return out;
}

export function utf8Bytes(s: string | undefined | null): number {
	return s ? new TextEncoder().encode(s).length : 0;
}

export interface ToolChip {
	durationMs?: number;
	running: boolean;
	argsBytes: number;
	resultBytes?: number;
	ok?: boolean;
	error?: string;
}

/**
 * A tool block's chip: the record's duration / sizes / status; while the call executes, the running
 * clock (now - start); without a record (older messages) the sizes of the stored args / result.
 */
export function toolChip(p: {
	record?: ChatMessageToolCallTiming;
	args?: string;
	result?: string;
	executingSince?: number | null;
	now?: number;
}): ToolChip {
	if (p.executingSince != null && !p.record) {
		return {
			argsBytes: utf8Bytes(p.args),
			durationMs: Math.max(0, (p.now ?? Date.now()) - p.executingSince),
			running: true
		};
	}

	const r = p.record;

	return {
		argsBytes: r?.args_bytes ?? utf8Bytes(p.args),
		durationMs: r?.duration_ms,
		error: r?.error,
		ok: r ? r.success : undefined,
		resultBytes: r?.result_bytes ?? (p.result !== undefined ? utf8Bytes(p.result) : undefined),
		running: false
	};
}

/** Context-bar line: the last request's prefill / decode speed and MTP %, the chat's compute (model + tools); the wall since the first message for the details. */
export interface SessionLine {
	prefillTps?: number;
	/** cache share of the last request's prompt */
	cacheShare?: number;
	decodeTps?: number;
	mtp?: number;
	/** compute of this chat: Σ prompt_ms + predicted_ms over all requests */
	modelMs: number;
	/** Σ tool-call durations */
	toolsMs: number;
	/** wall since the first message (details only) */
	sessionMs?: number;
}

export function sessionLine(messages: readonly DatabaseMessage[], now: number): SessionLine {
	const usage = cumulativeUsage(messages);
	const out: SessionLine = { modelMs: usage.promptMs + usage.outputMs, toolsMs: usage.toolsMs };

	for (let i = messages.length - 1; i >= 0; i--) {
		const t = messages[i].timings;

		if (messages[i].role !== MessageRole.ASSISTANT || !t) continue;

		if (t.prompt_n === undefined && t.predicted_n === undefined) continue;

		// the last request of the message (its own fields, not the flow totals)
		out.prefillTps = tps(t.prompt_n ?? 0, t.prompt_ms ?? 0);

		const all = (t.prompt_n ?? 0) + (t.cache_n ?? 0);

		if (all > 0) out.cacheShare = (t.cache_n ?? 0) / all;

		out.decodeTps = tps(t.predicted_n ?? 0, t.predicted_ms ?? 0);

		if ((t.draft_n ?? 0) > 0) out.mtp = (t.draft_n_accepted ?? 0) / (t.draft_n as number);

		break;
	}

	const first = messages.find((m) => m.type !== 'root' && m.role !== MessageRole.SYSTEM);

	if (first?.timestamp) out.sessionMs = Math.max(0, now - first.timestamp);

	return out;
}

/**
 * D4: prompt / cache / output totals across ALL turns of a chat. A turn = the messages after a user
 * message; an agentic turn counts its flow totals once (the last snapshot = the largest), a plain
 * turn its reply's timings.
 */
export function cumulativeUsage(messages: readonly DatabaseMessage[]): {
	read: number;
	cacheTotal: number;
	output: number;
	outputMs: number;
	promptMs: number;
	toolsMs: number;
} {
	const totals = { cacheTotal: 0, output: 0, outputMs: 0, promptMs: 0, read: 0, toolsMs: 0 };

	let flowTools = 0;
	let flow: ChatMessageAgenticLlmTotals | null = null;

	const closeFlow = () => {
		if (!flow) return;

		totals.read += flow.prompt_n ?? 0;
		totals.cacheTotal += flow.cache_n ?? 0;
		totals.output += flow.predicted_n ?? 0;
		totals.outputMs += flow.predicted_ms ?? 0;
		totals.promptMs += flow.prompt_ms ?? 0;
		totals.toolsMs += flowTools;
		flow = null;
		flowTools = 0;
	};

	for (const m of messages) {
		if (m.role === MessageRole.USER && !m.isSynthetic) {
			closeFlow();

			continue;
		}

		if (m.role !== MessageRole.ASSISTANT || !m.timings) continue;

		const a = m.timings.agentic?.llm;

		if (a) {
			// snapshots grow along the flow: keep the largest
			if (!flow || a.prompt_ms + a.predicted_ms >= flow.prompt_ms + flow.predicted_ms) {
				flow = a;
				flowTools = Math.max(flowTools, m.timings.agentic?.toolsMs ?? 0);
			}

			continue;
		}

		totals.read += m.timings.prompt_n ?? 0;
		totals.cacheTotal += m.timings.cache_n ?? 0;
		totals.output += m.timings.predicted_n ?? 0;
		totals.outputMs += m.timings.predicted_ms ?? 0;
		totals.promptMs += m.timings.prompt_ms ?? 0;
	}

	closeFlow();

	return totals;
}

export function formatDuration(ms: number | undefined): string {
	if (ms === undefined || !Number.isFinite(ms)) return '–';

	if (ms < 1000) return `${Math.round(ms)} ms`;

	if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;

	const s = Math.round(ms / 1000);
	const h = Math.floor(s / 3600);
	const m = Math.floor((s % 3600) / 60);

	return h > 0 ? `${h}h${String(m).padStart(2, '0')}m` : `${m}m${String(s % 60).padStart(2, '0')}s`;
}

/** m:ss / h:mm:ss */
export function formatClock(ms: number): string {
	const s = Math.max(0, Math.round(ms / 1000));
	const h = Math.floor(s / 3600);
	const m = Math.floor((s % 3600) / 60);
	const ss = String(s % 60).padStart(2, '0');

	return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
}

export function formatBytes(n: number | undefined): string {
	if (n === undefined) return '–';

	if (n < 1000) return `${n} B`;

	if (n < 1_000_000) return `${(n / 1000).toFixed(1)} kB`;

	return `${(n / 1_000_000).toFixed(1)} MB`;
}

export function formatTps(v: number | undefined): string {
	return v === undefined ? '–' : `${v >= 10 ? Math.round(v) : v.toFixed(1)} t/s`;
}

export function formatPct(v: number | undefined): string {
	return v === undefined ? '–' : `${Math.round(v * 100)} %`;
}

// ── W2.1: the collapsed surface ────────────────────────────────────────────────────────────────

/** The one muted chip line of a message: "12.3 s · 48 t/s · MTP 81 %" (parts it has). */
export function chipLine(t: ChatMessageTimings | undefined): string | null {
	const c = timingChips(t);

	if (!c) return null;

	const parts = [
		c.totalMs !== undefined ? formatDuration(c.totalMs) : '',
		c.decode?.tps !== undefined ? formatTps(c.decode.tps) : '',
		c.mtp !== undefined ? `MTP ${formatPct(c.mtp)}` : ''
	].filter(Boolean);

	return parts.length > 0 ? parts.join(' · ') : null;
}

/** The hover / expanded details of a message's chip line, incl. the turn summary. */
export function chipDetails(t: ChatMessageTimings | undefined): string[] {
	const c = timingChips(t);

	if (!c) return [];

	const k = (n: number) => n.toLocaleString();
	const lines: string[] = [];

	if (c.totalMs !== undefined || c.ttftMs !== undefined) {
		lines.push(`total ${formatDuration(c.totalMs)} · TTFT ${formatDuration(c.ttftMs)}`);
	}

	if (c.prefill) {
		lines.push(
			`prefill ${k(c.prefill.fresh)} fresh + ${k(c.prefill.cached)} cached (${formatPct(c.prefill.cacheShare)}) · ` +
				`${formatDuration(c.prefill.ms)} · ${formatTps(c.prefill.tps)}`
		);
	}

	if (c.decode) {
		lines.push(
			`decode ${k(c.decode.n)} tok · ${formatDuration(c.decode.ms)} · ${formatTps(c.decode.tps)}`
		);
	}

	if (c.mtp !== undefined) {
		const llm = llmTotals(t);

		lines.push(
			`MTP ${formatPct(c.mtp)} (${llm?.draft_n_accepted ?? 0} / ${llm?.draft_n ?? 0} drafted)`
		);
	}

	const turn = turnSummary(t);

	if (turn && t?.agentic) {
		lines.push(
			`turn ${formatDuration(turn.wallMs)} = model ${formatDuration(turn.modelMs)} + tools ` +
				`${formatDuration(turn.toolsMs)} + other ${formatDuration(turn.otherMs)} · ` +
				`${turn.llmCalls} LLM calls · ${turn.toolCalls} tools`
		);
	}

	return lines;
}

/** A tool block header fragment: "12.4 s · 18.2 kB" ("⏳ 4.1 s" while running). */
export function toolFragment(chip: ToolChip): string {
	if (chip.running) return `⏳ ${formatDuration(chip.durationMs)}`;

	return [
		chip.durationMs !== undefined ? formatDuration(chip.durationMs) : '',
		chip.resultBytes !== undefined ? formatBytes(chip.resultBytes) : ''
	]
		.filter(Boolean)
		.join(' · ');
}

/** The meter's "compact?" hint: from the fill threshold (%), else nothing. */
export function compactHint(usedPct: number, thresholdPct: number): boolean {
	return usedPct >= Math.min(100, Math.max(1, thresholdPct));
}

/** The context-bar popover rows (speed, cache, MTP, compute with the wall as a sub-line, reserve). */
export function popoverRows(p: {
	speed: SessionLine;
	reserve?: number;
	preTurnMin?: number;
	autoCompact: boolean;
}): { label: string; value: string; sub?: string }[] {
	const rows: { label: string; value: string; sub?: string }[] = [];
	const s = p.speed;

	if (s.prefillTps !== undefined || s.decodeTps !== undefined) {
		rows.push({
			label: 'Last request',
			value: `prefill ${formatTps(s.prefillTps)} · decode ${formatTps(s.decodeTps)}`
		});
	}

	if (s.cacheShare !== undefined)
		rows.push({ label: 'Cache share', value: formatPct(s.cacheShare) });

	if (s.mtp !== undefined) rows.push({ label: 'MTP acceptance', value: formatPct(s.mtp) });

	rows.push({
		label: 'Compute',
		sub:
			s.sessionMs !== undefined
				? `wall ${formatDuration(s.sessionMs)} since the first message`
				: undefined,
		value: `model ${formatClock(s.modelMs)} · tools ${formatClock(s.toolsMs)}`
	});

	if (p.reserve !== undefined && p.preTurnMin !== undefined) {
		rows.push({
			label: 'Reserve',
			value:
				`${p.reserve.toLocaleString()} tok · ` +
				(p.autoCompact
					? `compacts below ${p.preTurnMin.toLocaleString()} free`
					: 'auto-compaction off')
		});
	}

	return rows;
}
