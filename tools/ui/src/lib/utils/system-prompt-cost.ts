/**
 * What changing the system message of a conversation costs: the system prompt is the start of
 * every request, so the server has to re-process the whole conversation from scratch.
 */

import { MessageRole } from '$lib/enums';

export interface ReprocessCost {
	tokens: number;
	seconds: number;
}

// prefill speed when the conversation has no measured prefill of its own (a slow GPU at depth)
const FALLBACK_TOKENS_PER_SECOND = 300;

// below this a re-process takes seconds and needs no warning
export const REPROCESS_WARN_TOKENS = 4000;

/**
 * The conversation's size from the largest context a reply reported (prompt + cached + generated
 * tokens), else ~4 characters per token; the speed from the largest prefill the server measured in
 * this conversation. Null for a conversation without user messages.
 */
export function estimateReprocessCost(messages: readonly DatabaseMessage[]): ReprocessCost | null {
	if (!messages.some((m) => m.role === MessageRole.USER)) return null;

	let tokens = 0;
	let rate = 0;
	let rateTokens = 0;

	for (const m of messages) {
		const t = m.timings;

		if (!t) continue;

		tokens = Math.max(tokens, (t.prompt_n ?? 0) + (t.cache_n ?? 0) + (t.predicted_n ?? 0));

		if ((t.prompt_n ?? 0) >= 2000 && (t.prompt_ms ?? 0) > 0 && (t.prompt_n ?? 0) > rateTokens) {
			rateTokens = t.prompt_n ?? 0;
			rate = (t.prompt_n ?? 0) / ((t.prompt_ms ?? 1) / 1000);
		}
	}

	if (tokens === 0) {
		const chars = messages.reduce((n, m) => n + (m.content?.length ?? 0), 0);

		tokens = Math.round(chars / 4);
	}

	return { seconds: tokens / (rate > 0 ? rate : FALLBACK_TOKENS_PER_SECOND), tokens };
}

export function formatReprocessCost(cost: ReprocessCost): string {
	const tokens = cost.tokens >= 1000 ? `${Math.round(cost.tokens / 1000)}k` : `${cost.tokens}`;
	const time =
		cost.seconds >= 90
			? `~${Math.round(cost.seconds / 60)} min`
			: `~${Math.max(1, Math.round(cost.seconds))} s`;

	return `~${tokens} tokens, ${time}`;
}
