/**
 * Context reserve + auto-compaction (W1): pure arithmetic over a conversation path.
 *
 * The context in use is the server's own count where one exists (the last reply's
 * prompt_n + cache_n + predicted_n), plus a character estimate for what came after it.
 * A fixed token RESERVE stays free for the summary request: normal replies are capped at
 * free - reserve, and a turn that would not leave the reserve free compacts first.
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
	reserve: number;
	autoCompact: boolean;
	keepMessages: number;
	summaryPrompt: string;
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

function num(value: unknown, fallback: number): number {
	const n = Number(value);

	return Number.isFinite(n) && value !== '' && value !== null && value !== undefined ? n : fallback;
}

export function readContextReserveConfig(config: Record<string, unknown>): ContextReserveConfig {
	const prompt = typeof config.compactSummaryPrompt === 'string' ? config.compactSummaryPrompt : '';

	return {
		autoCompact: config.autoCompact !== false,
		keepMessages: Math.max(
			0,
			Math.floor(num(config.compactKeepMessages, CONTEXT_RESERVE.DEFAULT_KEEP_MESSAGES))
		),
		reserve: Math.max(
			0,
			Math.floor(num(config.contextReserveTokens, CONTEXT_RESERVE.DEFAULT_RESERVE_TOKENS))
		),
		summaryPrompt: prompt.trim() ? prompt : CONTEXT_RESERVE.DEFAULT_SUMMARY_PROMPT
	};
}

/**
 * The Context settings that apply to a conversation: its own override per field (W1.1), else the
 * global setting. Invalid override values (negative, non-finite, blank prompt) fall back too.
 */
export function resolveContextReserveConfig(
	config: Record<string, unknown>,
	overrides?: ConversationContextSettings | null
): ContextReserveConfig {
	const global = readContextReserveConfig(config);
	const o = overrides ?? {};
	const int = (v: unknown): number | undefined =>
		typeof v === 'number' && Number.isFinite(v) && v >= 0 ? Math.floor(v) : undefined;

	return {
		autoCompact: typeof o.autoCompact === 'boolean' ? o.autoCompact : global.autoCompact,
		keepMessages: int(o.keepMessages) ?? global.keepMessages,
		reserve: int(o.reserve) ?? global.reserve,
		summaryPrompt:
			typeof o.summaryPrompt === 'string' && o.summaryPrompt.trim()
				? o.summaryPrompt
				: global.summaryPrompt
	};
}

/** Overrides without the fields that are unset (undefined = no override at all). */
export function normalizeContextOverrides(
	o: ConversationContextSettings | null | undefined
): ConversationContextSettings | undefined {
	if (!o) return undefined;

	const out: ConversationContextSettings = {};

	if (typeof o.autoCompact === 'boolean') out.autoCompact = o.autoCompact;

	if (typeof o.keepMessages === 'number' && o.keepMessages >= 0) out.keepMessages = o.keepMessages;

	if (typeof o.reserve === 'number' && o.reserve >= 0) out.reserve = o.reserve;

	if (typeof o.summaryPrompt === 'string' && o.summaryPrompt.trim()) {
		out.summaryPrompt = o.summaryPrompt;
	}

	return Object.keys(out).length > 0 ? out : undefined;
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

/** reserve - instruction must hold a minimal summary; else the reserve is too small to compact. */
export function reserveHoldsSummary(reserve: number, instructionTokens: number): boolean {
	return reserve - instructionTokens >= CONTEXT_RESERVE.MIN_SUMMARY_TOKENS;
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
		return 'the summary hit its length limit (raise the context reserve)';
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

/** A turn of `pending` new tokens would leave less than the reserve (+ a minimal reply) free. */
export function shouldCompact(p: {
	nCtx: number | null;
	used: number;
	reserve: number;
	pending?: number;
}): boolean {
	if (!p.nCtx || p.reserve <= 0 || p.used <= 0) return false;

	return p.nCtx - p.used - (p.pending ?? 0) < p.reserve + CONTEXT_RESERVE.MIN_REPLY_TOKENS;
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
 * Null also when reserve - instruction cannot hold a minimal summary.
 */
export function planSummaryRequest(p: {
	messages: readonly DatabaseMessage[];
	nCtx: number;
	reserve: number;
	instructionTokens: number;
}): SummaryPlan | null {
	const { instructionTokens, messages, nCtx } = p;
	const desired = p.reserve - instructionTokens;

	if (desired < CONTEXT_RESERVE.MIN_SUMMARY_TOKENS) return null;

	const usage = measureContextUsage(messages, nCtx);
	const candidates: { keep: number; tokens: number; cut: boolean }[] = [];

	if (!usage.full) candidates.push({ cut: false, keep: messages.length, tokens: usage.used });

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
 * The last `keep` messages of the conversation body (system / cwd rows excluded) for the new
 * chat. Never starts with a tool result (its call goes along), and stays within `budget`
 * estimated tokens by dropping from the front.
 */
export function selectTailMessages(
	body: readonly DatabaseMessage[],
	keep: number,
	budget: number
): DatabaseMessage[] {
	if (keep <= 0 || body.length === 0) return [];

	let start = Math.max(0, body.length - keep);

	while (start > 0 && body[start].role === MessageRole.TOOL) start--;

	const cost = body.map(estimateMessageTokens);

	let total = cost.slice(start).reduce((a, b) => a + b, 0);

	const advance = () => {
		total -= cost[start];
		start++;
	};

	while (start < body.length && total > budget) {
		advance();

		while (start < body.length && body[start].role === MessageRole.TOOL) advance();
	}

	return body.slice(start);
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
