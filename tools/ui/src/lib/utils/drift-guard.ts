/**
 * Drift guard, webui half (spec r2 S2, v1 = containment only).
 *
 * The fork server stops a reply at a duplicate `</think>` in the content phase and reports
 * `finish_reason: "stop"` + `stop_type: "think_reenter"`; the suppressed closure is not emitted.
 * The webui then runs ONE bounded continuation of the same assistant message from that point:
 * an explicit content-phase continuation (`continue_final_message: "content"` - AUTO would resume
 * reasoning when the content is empty), the model card's Instruct sampling for that request only,
 * and a strictly positive remaining budget (max_tokens 0 means unlimited on the wire).
 */

import { CONTEXT_RESERVE } from '$lib/constants/context-reserve.constants';
import {
	presetValues,
	type SamplingPresetSettings
} from '$lib/constants/reasoning-effort.constants';

export const STOP_TYPE_THINK_REENTER = 'think_reenter';

export const DRIFT_GUARD_NOTES = {
	AGAIN: 'cut at a duplicate </think> again; kept as is (one retry per message)',
	CONTINUED: 'cut at a duplicate </think>; continued with conservative sampling',
	CUT_AGAIN:
		'cut at a duplicate </think>; continued with conservative sampling; cut again, kept as is',
	CUT_BUDGET:
		'cut at a duplicate </think>; continued with conservative sampling; cut again: budget',
	FAILED: (why: string) =>
		`cut at a duplicate </think>; the continuation failed (${why}), kept as is`,
	SKIPPED: (why: string) => `cut at a duplicate </think>; not continued: ${why}`
} as const;

/** The drift-guard state of a reply (persisted on the message). */
export interface DriftGuardState {
	/** the one continuation of this logical message was started (never a second one) */
	retried: boolean;
	/** the visible marker */
	note: string;
}

/**
 * The output budget of the one continuation: the budget of the request that stopped minus what
 * it generated (predicted_n: reasoning + content; the suppressed `</think>` is not counted by the
 * server, H6 round 2 - it is not charged here either). No budget
 * (unlimited) = the bounded fallback of the other auto-continuations. Never 0 or negative: a
 * zero max_tokens is sent as -1 (unlimited) by ChatService.
 */
export function driftContinuationBudget(p: {
	budget?: number;
	generated?: number;
}): { maxTokens: number } | { skip: string } {
	if (p.budget === undefined || !Number.isFinite(p.budget) || p.budget <= 0) {
		return { maxTokens: CONTEXT_RESERVE.GUARD_MAX_TOKENS };
	}

	if (p.generated === undefined || !Number.isFinite(p.generated) || p.generated < 0) {
		return { skip: 'the generated token count is unknown' };
	}

	const remaining = Math.floor(p.budget - p.generated);

	return remaining > 0 ? { maxTokens: remaining } : { skip: 'no output budget left' };
}

/**
 * The Instruct preset as Settings resolve it (the model card 0.7 / 0.8 / top-k 20 / min-p 0 /
 * presence 1.5 / repeat 1.0, with the user's edits of that preset).
 */
export function driftContinuationSampling(
	settings?: SamplingPresetSettings
): Record<string, number> {
	return { ...presetValues('instruct', settings).values };
}

/**
 * Whether a finished reply gets the drift-guard continuation: a `stop` with stop_type
 * think_reenter that has not had its one retry yet.
 */
export function driftGuardDecision(p: {
	finish?: string;
	stopType?: string;
	state?: DriftGuardState;
}): 'none' | 'continue' | 'again' {
	if (p.finish !== 'stop' || p.stopType !== STOP_TYPE_THINK_REENTER) return 'none';

	return p.state?.retried ? 'again' : 'continue';
}
