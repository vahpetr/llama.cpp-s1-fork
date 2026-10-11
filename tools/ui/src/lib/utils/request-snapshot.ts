/**
 * Per-turn snapshot of a request's template options and sampling (astra 0220 review, Major 3):
 * the native reasoning effort lives in the prompt's system block, so a continuation built from
 * the CURRENT picker state after a level change would re-process the prompt prefix. A reply's
 * continuations reuse the options of the request that produced it, verbatim.
 */

import type { RequestTemplateSnapshot } from '$lib/types';

export const SNAPSHOT_SAMPLING_KEYS = [
	'temperature',
	'top_p',
	'top_k',
	'min_p',
	'presence_penalty',
	'repeat_penalty'
] as const;

const TEMPLATE_KEYS = ['enableThinking', 'reasoningEffort', 'reasoningEffortNative'] as const;

/** The template options + sampling of request options (the shape getApiOptions returns). */
export function snapshotRequestOptions(options: Record<string, unknown>): RequestTemplateSnapshot {
	const sampling: Record<string, number> = {};

	for (const k of SNAPSHOT_SAMPLING_KEYS) {
		const v = options[k];

		if (typeof v === 'number' && Number.isFinite(v)) sampling[k] = v;
	}

	const out: RequestTemplateSnapshot = { sampling };

	if (typeof options.enableThinking === 'boolean') out.enableThinking = options.enableThinking;

	if (typeof options.reasoningEffort === 'string') out.reasoningEffort = options.reasoningEffort;

	if (typeof options.reasoningEffortNative === 'string') {
		out.reasoningEffortNative = options.reasoningEffortNative;
	}

	return out;
}

/**
 * Request options with a snapshot's template options and sampling put back exactly (a key the
 * snapshot does not have is removed, so the current state cannot add it). No snapshot = as is.
 */
export function applyRequestSnapshot(
	options: Record<string, unknown>,
	snapshot: RequestTemplateSnapshot | undefined
): Record<string, unknown> {
	if (!snapshot) return options;

	const out: Record<string, unknown> = { ...options };

	for (const k of TEMPLATE_KEYS) delete out[k];

	for (const k of SNAPSHOT_SAMPLING_KEYS) delete out[k];

	if (snapshot.enableThinking !== undefined) out.enableThinking = snapshot.enableThinking;

	if (snapshot.reasoningEffort !== undefined) out.reasoningEffort = snapshot.reasoningEffort;

	if (snapshot.reasoningEffortNative !== undefined) {
		out.reasoningEffortNative = snapshot.reasoningEffortNative;
	}

	Object.assign(out, snapshot.sampling);

	return out;
}
