/**
 * Context reserve + auto-compaction (W1, W1.2 = the pi brain rules): pure arithmetic over a
 * conversation path.
 *
 * The context in use is the server's own count where one exists (the last reply's
 * prompt_n + cache_n + predicted_n), plus a character estimate for what came after it.
 * The RESERVE is derived (contextBudget): normal replies are capped at free - reserve, a user turn
 * with less than reserve + 32768 free compacts first, a reply cut by that cap is continued after
 * a compaction, and a reply that stopped with an obviously unfinished tail gets one more go.
 */

import { CONTEXT_RESERVE } from '$lib/constants';
import { MessageRole, MessageType } from '$lib/enums';
import type {
	ApiChatMessageData,
	ChatMessageTimings,
	ConversationContextSettings,
	DatabaseMessage
} from '$lib/types';

export interface ContextReserveConfig {
	autoCompact: boolean;
	summaryPrompt: string;
	/** summary length limit (tokens) */
	summaryTokens: number;
	/** tokens of whole recent messages the continuation keeps verbatim */
	tailTokens: number;
}

export interface ContextUsage {
	/** tokens in use: the last measurement plus the estimate of what followed it */
	used: number;
	/** the server-measured part (0 = nothing measured) */
	measured: number;
	/** the measured message is the last one and it filled the context (a truncated reply) */
	full: boolean;
}

export interface SummaryPlan {
	/** number of path messages the summary request sends (a prefix of the path) */
	keep: number;
	/** tokens of that prefix */
	promptTokens: number;
	/** max_tokens of the summary request */
	maxTokens: number;
	/** the tail of the last turn was dropped (the context was full or had no room) */
	cut: boolean;
}

/** W1.2: the budget derived from the context length and the compaction prompt (as in pi). */
export interface ContextBudget {
	/** summary length actually used (the setting, at most n_ctx / 8) */
	summaryTokens: number;
	/** the instruction message with {{MAX_TOKENS}} filled in */
	instructionTokens: number;
	/** estimate slack: 10 % of the context in use, at least 2048 (n_ctx / 16 on small contexts) */
	slackTokens: number;
	/** kept free after every reply: summary + instruction + slack */
	reserve: number;
	/** a user turn starts only with at least this much free: reserve + 32768 (n_ctx / 4 if smaller) */
	preTurnMin: number;
}

function num(value: unknown, fallback: number): number {
	const n = Number(value);

	return Number.isFinite(n) && value !== '' && value !== null && value !== undefined ? n : fallback;
}

const positiveInt = (v: unknown): number | undefined =>
	typeof v === 'number' && Number.isFinite(v) && v > 0 ? Math.floor(v) : undefined;

export function readContextReserveConfig(config: Record<string, unknown>): ContextReserveConfig {
	const prompt = typeof config.compactSummaryPrompt === 'string' ? config.compactSummaryPrompt : '';
	const summary = Math.floor(
		num(config.compactSummaryTokens, CONTEXT_RESERVE.DEFAULT_SUMMARY_TOKENS)
	);
	const tail = Math.floor(num(config.compactTailTokens, CONTEXT_RESERVE.DEFAULT_TAIL_TOKENS));

	return {
		autoCompact: config.autoCompact !== false,
		summaryPrompt: prompt.trim() ? prompt : CONTEXT_RESERVE.DEFAULT_SUMMARY_PROMPT,
		summaryTokens: summary > 0 ? summary : CONTEXT_RESERVE.DEFAULT_SUMMARY_TOKENS,
		tailTokens: tail >= 0 ? tail : CONTEXT_RESERVE.DEFAULT_TAIL_TOKENS
	};
}

/**
 * The Context settings that apply to a conversation: its own override per field (W1.1), else the
 * global setting. Invalid override values (non-positive, non-finite, blank prompt) fall back too;
 * the W1/W1.1 fields `reserve` / `keepMessages` are no longer read (see legacyContextFields).
 */
export function resolveContextReserveConfig(
	config: Record<string, unknown>,
	overrides?: ConversationContextSettings | null
): ContextReserveConfig {
	const global = readContextReserveConfig(config);
	const o = overrides ?? {};
	const tail =
		typeof o.tailTokens === 'number' && Number.isFinite(o.tailTokens) && o.tailTokens >= 0
			? Math.floor(o.tailTokens)
			: undefined;

	return {
		autoCompact: typeof o.autoCompact === 'boolean' ? o.autoCompact : global.autoCompact,
		summaryPrompt:
			typeof o.summaryPrompt === 'string' && o.summaryPrompt.trim()
				? o.summaryPrompt
				: global.summaryPrompt,
		summaryTokens: positiveInt(o.summaryTokens) ?? global.summaryTokens,
		tailTokens: tail ?? global.tailTokens
	};
}

/**
 * Settings of W1/W1.1 that W1.2 no longer reads (the user-facing reserve and "messages kept"),
 * in the global config or a chat's overrides; they are ignored with a one-time notice.
 */
export function legacyContextFields(
	config: Record<string, unknown>,
	overrides?: ConversationContextSettings | null
): string[] {
	const out: string[] = [];
	const o = (overrides ?? {}) as Record<string, unknown>;

	if (config.contextReserveTokens !== undefined || o.reserve !== undefined) out.push('reserve');

	if (config.compactKeepMessages !== undefined || o.keepMessages !== undefined) {
		out.push('messages kept');
	}

	return out;
}

/** Overrides without the fields that are unset (undefined = no override at all); legacy fields dropped. */
export function normalizeContextOverrides(
	o: ConversationContextSettings | null | undefined
): ConversationContextSettings | undefined {
	if (!o) return undefined;

	const out: ConversationContextSettings = {};

	if (typeof o.autoCompact === 'boolean') out.autoCompact = o.autoCompact;

	if (positiveInt(o.summaryTokens) !== undefined) out.summaryTokens = positiveInt(o.summaryTokens);

	if (typeof o.tailTokens === 'number' && Number.isFinite(o.tailTokens) && o.tailTokens >= 0) {
		out.tailTokens = Math.floor(o.tailTokens);
	}

	if (typeof o.summaryPrompt === 'string' && o.summaryPrompt.trim()) {
		out.summaryPrompt = o.summaryPrompt;
	}

	return Object.keys(out).length > 0 ? out : undefined;
}

/**
 * W1.2 (the pi brain rules): reserve = summary + instruction + slack (10 % of used, min 2048);
 * reply cap = min(configured, n_ctx - used - reserve); a user turn needs reserve + 32768 free
 * (pre-turn minimum), else it compacts first. On small contexts the summary is at most n_ctx/8,
 * the slack floor n_ctx/16 and the pre-turn reply room n_ctx/4.
 */
export function contextBudget(p: {
	nCtx: number;
	used: number;
	summaryTokens: number;
	summaryPrompt: string;
}): ContextBudget {
	const summaryTokens = Math.max(1, Math.min(p.summaryTokens, Math.floor(p.nCtx / 8)));
	const instructionTokens =
		estimateTextTokens(buildSummaryInstruction(p.summaryPrompt, summaryTokens)) +
		CONTEXT_RESERVE.MESSAGE_OVERHEAD_TOKENS * 2;
	const slackTokens = Math.max(
		Math.min(CONTEXT_RESERVE.SLACK_MIN_TOKENS, Math.floor(p.nCtx / 16)),
		Math.ceil(Math.max(0, p.used) * CONTEXT_RESERVE.SLACK_SHARE)
	);
	const reserve = summaryTokens + instructionTokens + slackTokens;

	return {
		instructionTokens,
		preTurnMin: reserve + Math.min(CONTEXT_RESERVE.PRE_TURN_REPLY_TOKENS, Math.floor(p.nCtx / 4)),
		reserve,
		slackTokens,
		summaryTokens
	};
}

/** Rule 1: a user turn of `pending` new tokens would start with less than the pre-turn minimum free. */
export function shouldCompactBeforeTurn(p: {
	nCtx: number | null;
	used: number;
	pending?: number;
	preTurnMin: number;
}): boolean {
	if (!p.nCtx || p.used <= 0) return false;

	return p.nCtx - p.used - (p.pending ?? 0) < p.preTurnMin;
}

export function estimateTextTokens(
	text: string | undefined | null,
	charsPerToken: number = CONTEXT_RESERVE.CHARS_PER_TOKEN
): number {
	return text ? Math.ceil(text.length / charsPerToken) : 0;
}

const withMargin = (tokens: number): number => Math.ceil(tokens * CONTEXT_RESERVE.ESTIMATE_MARGIN);
const charsPerTokenFor = (role: string): number =>
	role === MessageRole.TOOL
		? CONTEXT_RESERVE.TOOL_CHARS_PER_TOKEN
		: CONTEXT_RESERVE.CHARS_PER_TOKEN;

/** Estimated tokens of a stored message (+10 % margin; tool results at 2.5 chars/token). */
export function estimateMessageTokens(m: DatabaseMessage): number {
	const cpt = charsPerTokenFor(m.role);

	let tokens =
		CONTEXT_RESERVE.MESSAGE_OVERHEAD_TOKENS +
		estimateTextTokens(m.content, cpt) +
		estimateTextTokens(m.reasoningContent) +
		estimateTextTokens(m.toolCalls, CONTEXT_RESERVE.TOOL_CHARS_PER_TOKEN);

	for (const extra of m.extra ?? []) {
		const content = (extra as { content?: unknown }).content;

		tokens +=
			typeof content === 'string' ? estimateTextTokens(content, cpt) : CONTEXT_RESERVE.MEDIA_TOKENS;
	}

	return withMargin(tokens);
}

/** Estimated tokens of a request message (+10 % margin; tool results at 2.5 chars/token). */
export function estimateApiMessageTokens(m: ApiChatMessageData): number {
	const cpt = charsPerTokenFor(m.role);

	let tokens = CONTEXT_RESERVE.MESSAGE_OVERHEAD_TOKENS + estimateTextTokens(m.reasoning_content);

	if (typeof m.content === 'string') {
		tokens += estimateTextTokens(m.content, cpt);
	} else {
		for (const part of m.content ?? []) {
			tokens +=
				part.type === 'text'
					? estimateTextTokens((part as { text?: string }).text, cpt)
					: CONTEXT_RESERVE.MEDIA_TOKENS;
		}
	}

	if (m.tool_calls?.length) {
		tokens += estimateTextTokens(
			JSON.stringify(m.tool_calls),
			CONTEXT_RESERVE.TOOL_CHARS_PER_TOKEN
		);
	}

	return withMargin(tokens);
}

/** Estimated tokens of the tool definitions a request carries (rendered into the prompt). */
export function estimateToolsTokens(tools: readonly unknown[] | null | undefined): number {
	if (!tools || tools.length === 0) return 0;

	return withMargin(
		estimateTextTokens(JSON.stringify(tools), CONTEXT_RESERVE.TOOL_CHARS_PER_TOKEN)
	);
}

/** prompt + cached + generated tokens of one reply */
export function timingsTotal(t: ChatMessageTimings | undefined): number {
	return t ? (t.prompt_n ?? 0) + (t.cache_n ?? 0) + (t.predicted_n ?? 0) : 0;
}

/** Context in use for a path of messages (root excluded). */
export function measureContextUsage(
	messages: readonly DatabaseMessage[],
	nCtx: number | null
): ContextUsage {
	let measuredAt = -1;
	let measured = 0;

	for (let i = messages.length - 1; i >= 0; i--) {
		const m = messages[i];

		if (m.role !== MessageRole.ASSISTANT) continue;

		const total = timingsTotal(m.timings);

		if (total > 0) {
			measured = total;
			measuredAt = i;

			break;
		}
	}

	let used = measured;

	for (let i = measuredAt + 1; i < messages.length; i++) {
		if (messages[i].type === MessageType.ROOT) continue;

		used += estimateMessageTokens(messages[i]);
	}

	const full = nCtx !== null && measuredAt === messages.length - 1 && measured >= nCtx - 1;

	return { full, measured, used };
}

/**
 * max_tokens for a normal reply: the configured limit, capped at free - reserve, never below 1
 * (with the reserve used up a reply cannot eat into it; the send path compacts or refuses
 * before that). Undefined = the configured value as is (no context size, or the reserve is off).
 */
export function replyTokenCap(p: {
	nCtx: number | null;
	used: number;
	reserve: number;
	configured?: number;
}): number | undefined {
	const configured = p.configured && p.configured > 0 ? p.configured : undefined;

	if (!p.nCtx || p.reserve <= 0) return configured;

	const room = p.nCtx - p.used - p.reserve;

	if (room < 1) return 1;

	return configured ? Math.min(configured, room) : room;
}

/**
 * Thinking budget of a reply whose output is capped by the reserve: at most half the cap, none
 * below 2 x MIN_ANSWER_TOKENS, so thinking cannot use up the room the answer needs.
 */
export function cappedThinkingBudget(cap: number | undefined): number | undefined {
	if (cap === undefined) return undefined;

	return cap < 2 * CONTEXT_RESERVE.MIN_ANSWER_TOKENS ? 0 : Math.floor(cap / 2);
}

/** The summary length must allow a minimal summary. */
export function summaryLengthUsable(summaryTokens: number): boolean {
	return summaryTokens >= CONTEXT_RESERVE.MIN_SUMMARY_TOKENS;
}

/**
 * The W1 ceilings on a finished request body, after the custom JSON: the output limit over every
 * alias the server reads (max_tokens / max_completion_tokens / n_predict; -1 or absent = none)
 * and the thinking budget (thinking_budget_tokens / reasoning_budget_tokens).
 */
export function applyRequestCeilings(
	body: Record<string, unknown>,
	c: { maxTokens?: number; thinkingBudget?: number }
): void {
	const lowest = (keys: string[], ceiling: number): number => {
		let v = ceiling;

		for (const k of keys) {
			const n = Number(body[k]);

			if (body[k] !== undefined && body[k] !== null && Number.isFinite(n) && n >= 0) {
				v = Math.min(v, n);
			}
		}

		return v;
	};

	if (c.maxTokens !== undefined && c.maxTokens >= 0) {
		const keys = ['max_tokens', 'max_completion_tokens', 'n_predict'];
		const v = lowest(keys, c.maxTokens);

		delete body.max_completion_tokens;
		delete body.n_predict;
		body.max_tokens = v;
	}

	if (c.thinkingBudget !== undefined && c.thinkingBudget >= 0) {
		const keys = ['thinking_budget_tokens', 'reasoning_budget_tokens'];
		const v = lowest(keys, c.thinkingBudget);

		body.thinking_budget_tokens = v;
		body.reasoning_budget_tokens = v;
	}
}

/**
 * Why a summary response cannot be used, or null: it must be a complete text answer - the stream
 * ended ([DONE]) with finish_reason "stop", no tool calls, and at least MIN_SUMMARY_CHARS.
 */
export function summaryRejection(r: {
	completed: boolean;
	finishReason?: string;
	toolCalls?: string;
	text: string;
}): string | null {
	if (!r.completed) return 'the summary stream ended before completion';

	if (r.toolCalls && r.toolCalls !== '[]') return 'the model answered with a tool call';

	if (r.finishReason === 'length') {
		return 'the summary hit its length limit (raise the summary length)';
	}

	if (r.finishReason !== 'stop') return `the summary did not finish (${r.finishReason ?? 'none'})`;

	if (r.text.trim().length < CONTEXT_RESERVE.MIN_SUMMARY_CHARS) return 'the summary is too short';

	return null;
}

/**
 * Estimated-token budget of the tail copied into the new chat: a quarter of the context at most,
 * and never more than what is left after the reserve, the summary note, the head (system + cwd
 * rows) and a minimal reply.
 */
export function tailTokenBudget(p: {
	nCtx: number;
	reserve: number;
	summaryTokens: number;
	headTokens: number;
}): number {
	return Math.max(
		0,
		Math.min(
			Math.floor(p.nCtx * CONTEXT_RESERVE.TAIL_CONTEXT_SHARE),
			p.nCtx - p.reserve - p.summaryTokens - p.headTokens - CONTEXT_RESERVE.MIN_REPLY_TOKENS
		)
	);
}

/** Trailing assistant rows with nothing in them (a failed or cancelled request) are not sent. */
export function dropTrailingEmptyAssistants(
	messages: readonly DatabaseMessage[]
): DatabaseMessage[] {
	const out = [...messages];

	while (out.length > 0) {
		const last = out[out.length - 1];

		if (
			last.role !== MessageRole.ASSISTANT ||
			last.content?.trim() ||
			last.reasoningContent?.trim() ||
			last.toolCalls?.trim()
		) {
			break;
		}

		out.pop();
	}

	return out;
}

/**
 * What the summary request sends. The path as the chat sends it (a pure extension of the cached
 * prompt) when it leaves room for the summary; otherwise - the context is full (a truncated
 * reply) or nearly so - the path is cut before the last assistant reply(s) that do not fit, at
 * a point whose token count the server measured (that reply's prompt_n + cache_n), so the
 * server can rewind to a checkpoint. Reuse is best effort: with no usable checkpoint at or
 * before that point the server re-reads the prefix (the summary request's timings show it).
 * Null also when the summary length is below the minimum.
 */
export function planSummaryRequest(p: {
	messages: readonly DatabaseMessage[];
	nCtx: number;
	summaryTokens: number;
	instructionTokens: number;
	/** rule 2: the last reply is a partial continued later - never part of the summary request */
	cutBeforeLast?: boolean;
}): SummaryPlan | null {
	const { instructionTokens, messages, nCtx } = p;
	const desired = p.summaryTokens;

	if (desired < CONTEXT_RESERVE.MIN_SUMMARY_TOKENS) return null;

	const usage = measureContextUsage(messages, nCtx);
	const candidates: { keep: number; tokens: number; cut: boolean }[] = [];

	if (!usage.full && !p.cutBeforeLast) {
		candidates.push({ cut: false, keep: messages.length, tokens: usage.used });
	}

	// cut points: before an assistant reply, at its measured prompt size. A count that is not
	// below a later one is not a prefix measurement (an agentic flow stamps its last turn's
	// timings on its first message) and is skipped.
	let limit = Infinity;

	for (let i = messages.length - 1; i > 0; i--) {
		const m = messages[i];

		if (m.role !== MessageRole.ASSISTANT || !m.timings) continue;

		const before = (m.timings.prompt_n ?? 0) + (m.timings.cache_n ?? 0);

		if (before <= 0 || before >= limit) continue;

		limit = before;
		candidates.push({ cut: true, keep: i, tokens: before });
	}

	for (const c of candidates) {
		if (!messages.slice(0, c.keep).some((m) => m.role === MessageRole.USER)) continue;

		const room = nCtx - c.tokens - instructionTokens - 1;

		if (room >= CONTEXT_RESERVE.MIN_SUMMARY_TOKENS) {
			return {
				cut: c.cut,
				keep: c.keep,
				maxTokens: Math.min(desired, room),
				promptTokens: c.tokens
			};
		}
	}

	return null;
}

/**
 * W1.2: the newest whole messages of the conversation body (system / cwd rows excluded) whose
 * estimated tokens fit `budget`, for the new chat. Never starts with a tool result (its call is
 * not there any more), so leading tool results are dropped.
 */
export function selectTailMessages(
	body: readonly DatabaseMessage[],
	budget: number
): DatabaseMessage[] {
	let start = body.length;
	let total = 0;

	while (start > 0) {
		const cost = estimateMessageTokens(body[start - 1]);

		if (total + cost > budget) break;

		total += cost;
		start--;
	}

	while (start < body.length && body[start].role === MessageRole.TOOL) start++;

	return body.slice(start);
}

// ── rules 2 / 3: continuing a reply from its exact stop point (as pi's lib/reply-continue.ts) ──

/**
 * Join a partial reply and its continuation (continue_final_message). llama-server streams the
 * prefilled assistant text back before the new tokens (v54 live smoke F4): when the continuation
 * starts with the whole partial, that echo is dropped; otherwise (no or partial echo) the pieces
 * are joined as they are.
 */
export function joinContinuation(partial: string, continuation: string): string {
	if (partial && continuation.startsWith(partial)) return continuation;

	return partial + continuation;
}

/** What to show while a continuation streams: an echo still in progress shows the partial only. */
export function liveContinuation(partial: string, continuation: string): string {
	if (partial && partial.startsWith(continuation)) return partial;

	return joinContinuation(partial, continuation);
}

/** Join two pieces of one stream; when `b` starts by repeating >= minOverlap chars of `a`'s end, the repeat is dropped. */
export function joinWithOverlap(
	a: string,
	b: string,
	minOverlap = 24,
	maxScan = 4000
): { text: string; overlap: number } {
	const lim = Math.min(a.length, b.length, maxScan);

	for (let k = lim; k >= minOverlap; k--) {
		if (a.endsWith(b.slice(0, k))) return { overlap: k, text: a + b.slice(k) };
	}

	return { overlap: 0, text: a + b };
}

/**
 * Rule 3: why a reply that stopped on end-of-turn looks unfinished, or null. An odd number of ```
 * fence lines always counts; a tail ending in ':' / '—' / '–', an open table row or an empty list
 * item counts only when the visible reply is SHORT (< 60 tokens at 4 chars/token) - a long answer
 * ending in a colon is a legitimate prose ending.
 */
export function unfinishedTail(text: string): string | null {
	const fences = (text.match(/^[ \t]*```/gm) ?? []).length;

	if (fences % 2 === 1) return 'unclosed code fence';

	const t = text.trimEnd();

	if (!t) return null;

	if (Math.ceil(t.length / 4) >= CONTEXT_RESERVE.UNFINISHED_SHORT_TOKENS) return null;

	const last = t.slice(t.lastIndexOf('\n') + 1);

	if (/[:：]$/.test(t)) return "short reply ending in ':'";

	if (/[—–]$/.test(t)) return 'short reply ending in a dash';

	if (/^\s*\|/.test(last) && !/\|\s*$/.test(last)) return 'short reply ending in an open table row';

	if (/^\s*(?:[-*+]|\d+[.)])\s*$/.test(last)) return 'short reply ending in an empty list item';

	return null;
}

export function buildSummaryInstruction(template: string, maxTokens: number): string {
	return template.replaceAll('{{MAX_TOKENS}}', String(maxTokens));
}

export function compactedConversationName(name: string): string {
	const m = /^(.*) \(cont\.(?: (\d+))?\)$/.exec(name);

	if (!m) return `${name} (cont.)`;

	return `${m[1]} (cont. ${m[2] ? Number(m[2]) + 1 : 2})`;
}

export function formatTokenCount(n: number): string {
	if (n >= 100_000) return `${Math.round(n / 1000)}k`;

	if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;

	return String(Math.max(0, Math.round(n)));
}
